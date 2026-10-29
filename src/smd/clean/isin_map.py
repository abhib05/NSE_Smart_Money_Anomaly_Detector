"""Stable security ids from ISIN/symbol histories.

ISIN is the primary identifier, but NSE usually issues a **new ISIN on a face-value
split** (e.g. Greenply INE461C01012 -> INE461C01020 on 2016-01-06), and companies change
symbols (same ISIN). We therefore build a chain:

* each `(isin, symbol)` pair observed in the bhavcopy forms a *run* `[valid_from, valid_to]`;
* two ISINs are the same security if the same symbol hands over from one ISIN to the other
  within `max_gap` trading days (the old ISIN stops, the new one starts);
* the security id (`sid`, stored in the `isin` column of downstream tables) is the
  earliest ISIN of the chain. It is only a label - no future information enters any value.
"""

from __future__ import annotations

import pandas as pd

from smd.utils import get_logger

log = get_logger(__name__)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_isin_map(bhav: pd.DataFrame, trading_days: pd.DatetimeIndex, max_gap: int = 5) -> pd.DataFrame:
    """`bhav` needs columns date, symbol, isin. Returns isin_map(isin, symbol, valid_from, valid_to, sid)."""
    df = bhav[["date", "symbol", "isin"]].copy()
    df["isin"] = df["isin"].fillna("SYM:" + df["symbol"])
    runs = (
        df.groupby(["isin", "symbol"], sort=False)["date"].agg(valid_from="min", valid_to="max").reset_index()
    )
    pos = pd.Series(range(len(trading_days)), index=trading_days)
    runs["from_pos"] = runs["valid_from"].map(pos)
    runs["to_pos"] = runs["valid_to"].map(pos)

    uf = _UnionFind()
    for isin in runs["isin"]:
        uf.find(isin)
    links = 0
    for _, g in runs.sort_values(["symbol", "valid_from"]).groupby("symbol", sort=False):
        if len(g) < 2:
            continue
        rows = g.to_dict("records")
        for prev, nxt in zip(rows[:-1], rows[1:], strict=False):
            if prev["isin"] == nxt["isin"]:
                continue
            gap = nxt["from_pos"] - prev["to_pos"]
            if 0 < gap <= max_gap:
                uf.union(prev["isin"], nxt["isin"])
                links += 1
    log.info("isin_map: %d (isin,symbol) runs, %d ISIN-change links", len(runs), links)

    first_seen = runs.groupby("isin")["valid_from"].min()
    runs["root"] = runs["isin"].map(uf.find)
    # sid = earliest-listed ISIN in each chain
    root_to_sid = (
        runs.assign(first=runs["isin"].map(first_seen))
        .sort_values("first")
        .drop_duplicates("root")
        .set_index("root")["isin"]
    )
    runs["sid"] = runs["root"].map(root_to_sid)
    return runs[["isin", "symbol", "valid_from", "valid_to", "sid"]].sort_values(["sid", "valid_from"]).reset_index(drop=True)


def map_symbol_date_to_sid(df: pd.DataFrame, isin_map: pd.DataFrame, max_staleness_days: int = 10) -> pd.Series:
    """Map rows with (symbol, date) to a sid using the run valid on that date (or the most
    recent run of that symbol that ended at most `max_staleness_days` before)."""
    left = df[["symbol", "date"]].reset_index().sort_values("date")
    right = isin_map[["symbol", "valid_from", "valid_to", "sid"]].sort_values("valid_from")
    m = pd.merge_asof(left, right, left_on="date", right_on="valid_from", by="symbol", direction="backward")
    stale = m["date"] > m["valid_to"] + pd.Timedelta(days=max_staleness_days)
    m.loc[stale, "sid"] = None
    return m.set_index("index")["sid"].reindex(df.index)
