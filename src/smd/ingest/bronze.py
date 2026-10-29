"""Raw files -> validated bronze parquet tables (one per dataset) + trading calendar."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from smd.config import Config
from smd.ingest import parsers
from smd.ingest.download import Manifest
from smd.utils import get_logger, write_dataset
from smd.validation.schemas import validate

log = get_logger(__name__)

KEEP_SERIES = ("EQ", "BE")  # EQ is the study universe; BE (trade-for-trade) kept in bronze for reference


def _latest_ok(manifest: Manifest, dataset: str) -> pd.DataFrame:
    """OK entries; for periodic datasets keep only the newest file per key (as-of refreshes)."""
    e = manifest.entries(dataset)
    e = e[e["status"] == "ok"].sort_values("fetched_at")
    return e.drop_duplicates("key", keep="last")


def _parse_many(raw_dir: Path, entries: pd.DataFrame, fn: Callable[[bytes], pd.DataFrame]) -> pd.DataFrame:
    def one(relpath: str) -> pd.DataFrame | None:
        try:
            df = fn((raw_dir / relpath).read_bytes())
        except Exception as exc:  # noqa: BLE001 - a bad file is logged and skipped, not fatal
            log.warning("parse failed for %s: %s", relpath, exc)
            return None
        df["source_file"] = relpath
        return df

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [f for f in ex.map(one, entries["relpath"].tolist()) if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_bhavcopy(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    df = _parse_many(cfg.paths.raw, _latest_ok(manifest, "bhavcopy"), parsers.parse_bhavcopy)
    df = df[df["series"].isin(KEEP_SERIES)]
    df = df.dropna(subset=["close"]).drop_duplicates(["date", "symbol", "series"], keep="last")
    return validate(df.sort_values(["date", "symbol"]).reset_index(drop=True), "bronze.bhavcopy")


def build_delivery(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    df = _parse_many(cfg.paths.raw, _latest_ok(manifest, "delivery"), parsers.parse_delivery_mto)
    df = df[df["series"].isin(KEEP_SERIES)].drop_duplicates(["date", "symbol", "series"], keep="last")
    return validate(df.sort_values(["date", "symbol"]).reset_index(drop=True), "bronze.delivery")


def build_deals(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    frames = []
    for kind in ("bulk", "block"):
        f = _parse_many(cfg.paths.raw, _latest_ok(manifest, kind), partial(parsers.parse_deals_csv, deal_type=kind))
        if len(f):
            frames.append(f)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*parsers.DEAL_COLUMNS, "source_file"])
    df = df.dropna(subset=["qty", "price"]).drop_duplicates()
    return validate(df.sort_values(["date", "symbol"]).reset_index(drop=True), "bronze.deals")


def build_index(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    df = _parse_many(cfg.paths.raw, _latest_ok(manifest, "index"), parsers.parse_index_snapshot)
    df = df.dropna(subset=["close"]).drop_duplicates(["date", "index_name"], keep="last")
    return validate(df.sort_values(["date", "index_name"]).reset_index(drop=True), "bronze.index")


def build_corp_actions(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    df = _parse_many(cfg.paths.raw, _latest_ok(manifest, "corp_actions"), parsers.parse_corp_actions_json)
    df = df.drop_duplicates(["symbol", "isin", "ex_date", "subject"], keep="last")
    return validate(df.sort_values(["ex_date", "symbol"]).reset_index(drop=True), "bronze.corporate_actions")


def build_calendar(manifest: Manifest) -> pd.DataFrame:
    """Trading calendar: a weekday with a bhavcopy is a trading day. A weekday whose index
    snapshot exists but whose bhavcopy or delivery file is missing is a *gap*."""

    def ok_days(ds: str) -> set:
        e = manifest.entries(ds)
        return set(pd.to_datetime(e.loc[e["status"] == "ok", "key"]))

    bh, ix, dv = ok_days("bhavcopy"), ok_days("index"), ok_days("delivery")
    tried = set(pd.to_datetime(manifest.entries("bhavcopy")["key"]))
    days = sorted(tried | ix)
    cal = pd.DataFrame({"date": pd.to_datetime(days)})
    cal["has_bhavcopy"] = cal["date"].isin(bh)
    cal["has_index"] = cal["date"].isin(ix)
    cal["has_delivery"] = cal["date"].isin(dv)
    cal["is_trading_day"] = cal["has_bhavcopy"]
    cal["is_gap"] = (cal["has_index"] & ~cal["has_bhavcopy"]) | (cal["has_bhavcopy"] & ~cal["has_delivery"])
    return validate(cal, "bronze.calendar")


BUILDERS = {
    "bhavcopy": build_bhavcopy,
    "delivery": build_delivery,
    "deals": build_deals,
    "index": build_index,
    "corporate_actions": build_corp_actions,
}


def build_bronze(cfg: Config, tables: list[str] | None = None) -> dict[str, int]:
    manifest = Manifest(cfg.paths.raw)
    counts: dict[str, int] = {}
    for name, fn in BUILDERS.items():
        if tables and name not in tables:
            continue
        df = fn(cfg, manifest)
        write_dataset(cfg, "bronze", name, df)
        counts[name] = len(df)
        log.info("bronze.%s: %d rows", name, len(df))
    cal = build_calendar(manifest)
    write_dataset(cfg, "bronze", "calendar", cal)
    counts["calendar_trading_days"] = int(cal["is_trading_day"].sum())
    counts["calendar_gaps"] = int(cal["is_gap"].sum())
    return counts
