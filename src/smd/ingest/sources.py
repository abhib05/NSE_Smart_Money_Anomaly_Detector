"""Where each dataset lives on NSE and where its raw file is stored locally.

Datasets are either *daily* (one file per trading day) or *periodic* (one file per
month/quarter, fetched with a date-range query). Periodic files whose window is not
yet complete are stored with an `_asof<date>` suffix so a later, complete download
never overwrites an earlier raw file (raw data stays immutable).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import PurePosixPath

ARCH = "https://nsearchives.nseindia.com"
API = "https://www.nseindia.com/api"

DAILY_DATASETS = ("bhavcopy", "delivery", "index")
PERIODIC_DATASETS = ("bulk", "block", "corp_actions")
SNAPSHOT_DATASETS = ("symbolchange",)
ALL_DATASETS = DAILY_DATASETS + PERIODIC_DATASETS + SNAPSHOT_DATASETS


@dataclass(frozen=True)
class Request:
    dataset: str
    key: str  # unique id within dataset (date, month or quarter)
    url: str
    params: dict | None
    relpath: str  # path under data/raw
    window_end: date  # last date the file covers (for 404/as-of logic)


def bhavcopy_request(d: date, udiff_cutoff: date) -> Request:
    mon = d.strftime("%b").upper()
    if d < udiff_cutoff:
        name = f"cm{d:%d}{mon}{d.year}bhav.csv.zip"
        url = f"{ARCH}/content/historical/EQUITIES/{d.year}/{mon}/{name}"
    else:
        name = f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        url = f"{ARCH}/content/cm/{name}"
    return Request("bhavcopy", d.isoformat(), url, None, str(PurePosixPath("bhavcopy", str(d.year), name)), d)


def delivery_request(d: date) -> Request:
    name = f"MTO_{d:%d%m%Y}.DAT"
    return Request(
        "delivery", d.isoformat(), f"{ARCH}/archives/equities/mto/{name}", None,
        str(PurePosixPath("delivery", str(d.year), name)), d,
    )


def index_request(d: date) -> Request:
    name = f"ind_close_all_{d:%d%m%Y}.csv"
    return Request(
        "index", d.isoformat(), f"https://www.niftyindices.com/Daily_Snapshot/{name}", None,
        str(PurePosixPath("index", str(d.year), name)), d,
    )


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    out = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((cur, nxt - timedelta(days=1)))
        cur = nxt
    return out


def quarter_windows(start: date, end: date) -> list[tuple[date, date]]:
    out = []
    cur = date(start.year, 3 * ((start.month - 1) // 3) + 1, 1)
    while cur <= end:
        m = cur.month + 3
        nxt = date(cur.year + (m > 12), (m - 1) % 12 + 1, 1)
        out.append((cur, nxt - timedelta(days=1)))
        cur = nxt
    return out


def _asof_suffix(window_end: date, today: date) -> str:
    return "" if window_end < today else f"_asof{today:%Y%m%d}"


def deals_request(kind: str, w0: date, w1: date, today: date) -> Request:
    assert kind in ("bulk", "block")
    end = min(w1, today - timedelta(days=1))
    name = f"{kind}_{w0:%Y%m}{_asof_suffix(w1, today)}.csv"
    params = {"optionType": f"{kind}_deals", "from": f"{w0:%d-%m-%Y}", "to": f"{end:%d-%m-%Y}", "csv": "true"}
    return Request(
        kind, f"{w0:%Y-%m}", f"{API}/historicalOR/bulk-block-short-deals", params,
        str(PurePosixPath("deals", kind, name)), w1,
    )


def corp_actions_request(w0: date, w1: date, today: date) -> Request:
    q = (w0.month - 1) // 3 + 1
    name = f"ca_{w0.year}Q{q}{_asof_suffix(w1, today)}.json"
    params = {"index": "equities", "from_date": f"{w0:%d-%m-%Y}", "to_date": f"{w1:%d-%m-%Y}"}
    return Request(
        "corp_actions", f"{w0.year}Q{q}", f"{API}/corporates-corporateActions", params,
        str(PurePosixPath("corp_actions", name)), w1,
    )


def symbolchange_request(today: date) -> Request:
    name = f"symbolchange_asof{today:%Y%m%d}.csv"
    return Request(
        "symbolchange", today.isoformat(), f"{ARCH}/content/equities/symbolchange.csv", None,
        str(PurePosixPath("meta", name)), today,
    )


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out
