"""Point-in-time, survivorship-bias-free liquidity universe.

On every trading day *t*, using only rows dated <= *t*:
* eligible = traded on *t*, >= `min_history_days` rows of history, and <= `max_missing_frac`
  missing days in the last `liquidity_lookback_days` trading days;
* rank eligible stocks by trailing median traded value (₹) over that lookback;
* the top `size` are in the universe.

Stocks that later delist stay in the historical universe for the days they qualified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.config import UniverseCfg


def wide(panel: pd.DataFrame, col: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    w = panel.pivot(index="date", columns="isin", values=col)
    return w.reindex(days)


def compute_universe(panel: pd.DataFrame, days: pd.DatetimeIndex, cfg: UniverseCfg) -> pd.DataFrame:
    """Return long frame (date, isin, in_universe, liquidity_rank, med_traded_value) for eligible rows."""
    p = panel[["date", "isin", "traded_value", "dq_exclude"]].copy()
    p.loc[p["dq_exclude"], "traded_value"] = np.nan
    tv = wide(p, "traded_value", days)
    L = cfg.liquidity_lookback_days
    traded = tv.notna()
    present = traded.rolling(L, min_periods=L).sum()
    missing_frac = 1.0 - present / L
    history = traded.cumsum()
    med = tv.rolling(L, min_periods=max(1, int(np.ceil(L * (1 - cfg.max_missing_frac))))).median()
    eligible = traded & (history >= cfg.min_history_days) & (missing_frac <= cfg.max_missing_frac + 1e-12) & med.notna()
    med_e = med.where(eligible)
    rank = med_e.rank(axis=1, ascending=False, method="first")
    out = pd.DataFrame({
        "liquidity_rank": rank.stack(),
        "med_traded_value": med_e.stack(),
    }).reset_index()
    out.columns = ["date", "isin", "liquidity_rank", "med_traded_value"]
    out["in_universe"] = out["liquidity_rank"] <= cfg.size
    out["liquidity_rank"] = out["liquidity_rank"].astype(int)
    return out.sort_values(["date", "liquidity_rank"]).reset_index(drop=True)


def universe_size_by_date(universe: pd.DataFrame) -> pd.Series:
    return universe[universe["in_universe"]].groupby("date").size()
