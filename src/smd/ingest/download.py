"""Idempotent downloaders.

Every request outcome is recorded in `data/raw/_manifest.parquet`:
`dataset, key, url, relpath, status (ok|missing), http_status, sha256, bytes, fetched_at,
window_end`. A request is skipped when the manifest already holds:

* an `ok` entry whose file is on disk with the recorded checksum, and whose window was
  complete when fetched (periodic files fetched mid-window are refreshed once per day,
  into a new `_asof` file), or
* a `missing` (HTTP 404) entry fetched more than `refetch_recent_days` after the date -
  i.e. NSE had enough time to publish it, so the 404 is a holiday or a permanent gap.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from smd.config import Config
from smd.ingest import sources
from smd.ingest.http import ThrottledClient
from smd.ingest.sources import Request
from smd.utils import get_logger, sha256_bytes

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "dataset", "key", "url", "relpath", "status", "http_status", "sha256", "bytes", "fetched_at", "window_end",
]


class Manifest:
    def __init__(self, raw_dir: Path) -> None:
        self.path = raw_dir / "_manifest.parquet"
        if self.path.exists():
            self.df = pd.read_parquet(self.path)
        else:
            self.df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        self._pending: list[dict] = []

    def entries(self, dataset: str) -> pd.DataFrame:
        self.flush()
        return self.df[self.df["dataset"] == dataset]

    def add(self, row: dict) -> None:
        self._pending.append(row)
        if len(self._pending) >= 50:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        new = pd.DataFrame(self._pending, columns=MANIFEST_COLUMNS)
        self.df = new if self.df.empty else pd.concat([self.df, new], ignore_index=True)
        # keep the latest outcome for each (dataset, relpath)
        self.df = self.df.drop_duplicates(["dataset", "relpath"], keep="last").reset_index(drop=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        self.df.to_parquet(tmp, index=False)
        tmp.replace(self.path)
        self._pending = []


def validate_content(dataset: str, content: bytes) -> bool:
    """Cheap sanity check that a 200 response is the file we asked for, not an HTML error page."""
    if dataset == "bhavcopy":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                return any(n.lower().endswith(".csv") for n in zf.namelist())
        except zipfile.BadZipFile:
            return False
    if dataset == "delivery":
        return content.lstrip().startswith(b"Security Wise Delivery Position")
    if dataset == "index":
        return content.lstrip(b"\xef\xbb\xbf").startswith(b"Index Name")
    if dataset in ("bulk", "block"):
        head = content.lstrip(b"\xef\xbb\xbf")[:200].replace(b'"', b"")
        return head.startswith(b"Date")
    if dataset == "corp_actions":
        try:
            return isinstance(json.loads(content), list)
        except ValueError:
            return False
    if dataset == "symbolchange":
        return b"OLD SYMBOL" in content[:200].upper()
    return True


def should_skip(req: Request, existing: pd.DataFrame, raw_dir: Path, today: date, refetch_recent_days: int) -> bool:
    hit = existing[existing["relpath"] == req.relpath]
    if hit.empty:
        return False
    row = hit.iloc[-1]
    if row["status"] == "ok":
        return (raw_dir / req.relpath).exists()
    if row["status"] == "missing":
        fetched = pd.Timestamp(row["fetched_at"]).date()
        return (fetched - req.window_end).days > refetch_recent_days
    return False  # errors are retried


def fetch_all(
    requests: Iterable[Request],
    client: ThrottledClient,
    manifest: Manifest,
    raw_dir: Path,
    today: date,
    refetch_recent_days: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    reqs = list(requests)
    stats = {"requested": 0, "skipped": 0, "ok": 0, "missing": 0, "error": 0}
    by_ds: dict[str, pd.DataFrame] = {}
    for i, req in enumerate(reqs):
        if req.dataset not in by_ds:
            by_ds[req.dataset] = manifest.entries(req.dataset)
        if should_skip(req, by_ds[req.dataset], raw_dir, today, refetch_recent_days):
            stats["skipped"] += 1
            continue
        stats["requested"] += 1
        row = {
            "dataset": req.dataset, "key": req.key, "url": req.url, "relpath": req.relpath,
            "fetched_at": datetime.now(UTC).replace(tzinfo=None), "window_end": pd.Timestamp(req.window_end),
            "sha256": None, "bytes": 0,
        }
        try:
            res = client.get(req.url, req.params)
        except RuntimeError as exc:
            log.warning("error fetching %s %s: %s", req.dataset, req.key, exc)
            stats["error"] += 1
            manifest.add({**row, "status": "error", "http_status": -1})
            continue
        if res.status == 200 and validate_content(req.dataset, res.content):
            dest = raw_dir / req.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():  # immutable raw: never overwrite a different file
                if sha256_bytes(dest.read_bytes()) != sha256_bytes(res.content):
                    log.warning("refusing to overwrite raw file %s with different content", dest)
                    stats["error"] += 1
                    continue
            else:
                dest.write_bytes(res.content)
            manifest.add({**row, "status": "ok", "http_status": 200,
                          "sha256": sha256_bytes(res.content), "bytes": len(res.content)})
            stats["ok"] += 1
        elif res.status in (200, 404):
            # 404, or a 200 that is not the expected file (NSE serves HTML for missing files)
            manifest.add({**row, "status": "missing", "http_status": res.status})
            stats["missing"] += 1
        else:
            manifest.add({**row, "status": "error", "http_status": res.status})
            stats["error"] += 1
        if progress and (i % 100 == 0):
            progress(i, len(reqs))
    manifest.flush()
    return stats


def _progress_logger(ds: str) -> Callable[[int, int], None]:
    def report(i: int, n: int) -> None:
        log.info("  %s %d/%d", ds, i, n)

    return report


def build_requests(cfg: Config, dataset: str, manifest: Manifest, start: date, end: date, today: date) -> list[Request]:
    if dataset == "bhavcopy":
        return [sources.bhavcopy_request(d, cfg.ingest.udiff_cutoff) for d in sources.weekdays(start, end)]
    if dataset == "index":
        return [sources.index_request(d) for d in sources.weekdays(start, end)]
    if dataset == "delivery":
        # only days on which the exchange traded (bhavcopy or index exists)
        ok = pd.concat([manifest.entries("bhavcopy"), manifest.entries("index")])
        ok = ok[ok["status"] == "ok"]
        days = sorted({date.fromisoformat(k) for k in ok["key"]})
        days = [d for d in days if start <= d <= end]
        return [sources.delivery_request(d) for d in days]
    if dataset in ("bulk", "block"):
        return [sources.deals_request(dataset, w0, w1, today) for w0, w1 in sources.month_windows(start, end)]
    if dataset == "corp_actions":
        # corporate actions are announced ahead of the ex-date: fetch up to the end of the next quarter
        horizon = end + timedelta(days=92)
        return [sources.corp_actions_request(w0, w1, today) for w0, w1 in sources.quarter_windows(start, horizon)]
    if dataset == "symbolchange":
        return [sources.symbolchange_request(today)]
    raise ValueError(f"unknown dataset {dataset}")


def ingest(cfg: Config, datasets: list[str], start: date | None = None, end: date | None = None,
           client: ThrottledClient | None = None, today: date | None = None) -> dict[str, dict[str, int]]:
    today = today or date.today()
    start = start or cfg.ingest.start
    end = end or cfg.ingest.end or (today - timedelta(days=1))
    raw_dir = cfg.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(raw_dir)
    own_client = client is None
    client = client or ThrottledClient(cfg.ingest.rate_limit_per_sec, cfg.ingest.max_retries, cfg.ingest.timeout_sec)
    # order matters: delivery uses the bhavcopy/index outcomes to know which days traded
    order = [d for d in ("bhavcopy", "index", "delivery", "bulk", "block", "corp_actions", "symbolchange") if d in datasets]
    out: dict[str, dict[str, int]] = {}
    try:
        for ds in order:
            reqs = build_requests(cfg, ds, manifest, start, end, today)
            log.info("ingest %s: %d candidate files", ds, len(reqs))
            out[ds] = fetch_all(
                reqs, client, manifest, raw_dir, today, cfg.ingest.refetch_recent_days,
                progress=_progress_logger(ds),
            )
            log.info("ingest %s done: %s", ds, out[ds])
    finally:
        manifest.flush()
        if own_client:
            client.close()
    return out
