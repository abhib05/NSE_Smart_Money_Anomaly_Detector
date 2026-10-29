"""Rank information coefficient (IC) and quintile analysis.

* daily rank IC = Spearman correlation across stocks between a score at *t* and the
  forward excess return label at *t*;
* summary: mean IC, IC std, IC information ratio (mean/std), % of dates with IC > 0, and a
  **moving-block bootstrap** 95% CI for the mean IC (block = label horizon, because daily
  ICs of overlapping h-day labels are autocorrelated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(df: pd.DataFrame, score: str, target: str, min_names: int = 10) -> pd.Series:
    d = df[["date", score, target]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby("date")
    rs = g[score].rank()
    rt = g[target].rank()
    tmp = pd.DataFrame({"date": d["date"], "a": rs, "b": rt})
    n = tmp.groupby("date").size()
    tmp["a_c"] = tmp["a"] - tmp.groupby("date")["a"].transform("mean")
    tmp["b_c"] = tmp["b"] - tmp.groupby("date")["b"].transform("mean")
    cov = (tmp["a_c"] * tmp["b_c"]).groupby(tmp["date"]).sum()
    var = np.sqrt((tmp["a_c"] ** 2).groupby(tmp["date"]).sum() * (tmp["b_c"] ** 2).groupby(tmp["date"]).sum())
    ic = cov / var.replace(0, np.nan)
    return ic[n >= min_names].dropna()


def block_bootstrap_mean_ci(x: np.ndarray, block: int, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def ic_summary(ic: pd.Series, horizon: int, n_boot: int = 1000, seed: int = 0) -> dict:
    if ic.empty:
        return {"n_dates": 0}
    lo, hi = block_bootstrap_mean_ci(ic.to_numpy(), horizon, n_boot, seed)
    sd = float(ic.std())
    return {
        "n_dates": int(len(ic)),
        "mean_ic": float(ic.mean()),
        "ic_std": sd,
        "ic_ir": float(ic.mean() / sd) if sd > 0 else np.nan,
        "pct_positive": float((ic > 0).mean()),
        "ci_low": lo,
        "ci_high": hi,
    }


def quintile_returns(df: pd.DataFrame, score: str, target: str, q: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Mean target by score quintile per date. Returns (per-date table, top-minus-bottom spread series)."""
    d = df[["date", score, target]].dropna().copy()
    pct = d.groupby("date")[score].rank(pct=True, method="first")
    d["q"] = np.ceil(pct * q).clip(1, q).astype(int)
    tab = d.groupby(["date", "q"])[target].mean().unstack("q")
    spread = (tab[q] - tab[1]).dropna() if q in tab and 1 in tab else pd.Series(dtype=float)
    return tab, spread
