"""Feature context: wide (date x security) matrices built from the silver panel.

All features are computed on these matrices with *trailing* windows only, so a value at
row *t* depends on rows <= *t*. Rows flagged `dq_exclude` are blanked (NaN) first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WIDE_COLUMNS = {
    "o": "adj_open", "h": "adj_high", "l": "adj_low", "c": "adj_close",
    "v": "adj_volume", "tv": "traded_value", "dqty": "adj_deliverable_qty", "dpct": "delivery_pct",
}


@dataclass
class FeatureContext:
    days: pd.DatetimeIndex
    m: dict[str, pd.DataFrame]          # wide matrices
    deals: pd.DataFrame                 # silver deals restricted to the securities in `m`
    universe_mask: pd.DataFrame         # bool wide: in universe on that date
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, k: str) -> pd.DataFrame:
        return self.m[k]

    def memo(self, key: str, fn) -> pd.DataFrame:
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    # --- shared building blocks -------------------------------------------------------
    @property
    def c_ff(self) -> pd.DataFrame:
        """Adjusted close forward-filled over short gaps (<= 5 days) for multi-day returns."""
        return self.memo("c_ff", lambda: self["c"].ffill(limit=5))

    @property
    def ret1(self) -> pd.DataFrame:
        return self.memo("ret1", lambda: self["c"] / self.c_ff.shift(1) - 1.0)

    @property
    def mkt_ret(self) -> pd.Series:
        """Equal-weight daily return of the point-in-time universe."""
        return self.memo("mkt", lambda: self.ret1.where(self.universe_mask).mean(axis=1).to_frame("m"))["m"]


def build_context(panel: pd.DataFrame, deals: pd.DataFrame, universe: pd.DataFrame,
                  days: pd.DatetimeIndex, securities: list[str] | None = None) -> FeatureContext:
    p = panel
    if securities is not None:
        p = p[p["isin"].isin(securities)]
    p = p.copy()
    excl = p["dq_exclude"].to_numpy()
    m: dict[str, pd.DataFrame] = {}
    for key, col in WIDE_COLUMNS.items():
        vals = p[col].astype(float).to_numpy().copy()
        vals[excl] = np.nan
        p["_v"] = vals
        m[key] = p.pivot(index="date", columns="isin", values="_v").reindex(days)
    cols = m["c"].columns
    for k in m:
        m[k] = m[k].reindex(columns=cols)
    u = universe[universe["in_universe"]]
    mask = pd.DataFrame(False, index=days, columns=cols)
    uu = u[u["isin"].isin(cols)]
    if len(uu):
        r = days.get_indexer(uu["date"])
        cidx = cols.get_indexer(uu["isin"])
        arr = mask.to_numpy()
        arr[r, cidx] = True
        mask = pd.DataFrame(arr, index=days, columns=cols)
    d = deals[deals["isin"].isin(cols)].copy()
    return FeatureContext(days=days, m=m, deals=d, universe_mask=mask)


# --- helpers ------------------------------------------------------------------------------

def safe_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def trailing_z(x: pd.DataFrame, n: int, min_periods: int | None = None) -> pd.DataFrame:
    """z-score of x_t versus the previous `n` observations (t-n .. t-1)."""
    mp = min_periods or max(2, n // 2)
    prev = x.shift(1)
    mu = prev.rolling(n, min_periods=mp).mean()
    sd = prev.rolling(n, min_periods=mp).std()
    return (x - mu) / sd.where(sd > 1e-12)


def rolling_beta(r: pd.DataFrame, m: pd.Series, n: int, min_periods: int) -> pd.DataFrame:
    valid = r.notna()
    mm = pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    mm = mm.where(valid)
    rr = r.where(mm.notna())
    e_rm = (rr * mm).rolling(n, min_periods=min_periods).mean()
    e_r = rr.rolling(n, min_periods=min_periods).mean()
    e_m = mm.rolling(n, min_periods=min_periods).mean()
    e_mm = (mm * mm).rolling(n, min_periods=min_periods).mean()
    var = e_mm - e_m**2
    return (e_rm - e_r * e_m) / var.where(var > 1e-14)
