"""Performance metrics, incl. the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

PERIODS = 252
EULER_GAMMA = 0.5772156649


def cagr(r: pd.Series, periods: int = PERIODS) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    growth = float((1 + r).prod())
    return growth ** (periods / len(r)) - 1 if growth > 0 else -1.0


def ann_vol(r: pd.Series, periods: int = PERIODS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    ex = r.dropna() - rf_annual / periods
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    ex = r.dropna() - rf_annual / periods
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min()) if len(r) else np.nan


def calmar(r: pd.Series, periods: int = PERIODS) -> float:
    mdd = max_drawdown(r)
    return float(cagr(r, periods) / abs(mdd)) if mdd < 0 else np.nan


def beta_alpha(r: pd.Series, b: pd.Series, periods: int = PERIODS) -> tuple[float, float]:
    d = pd.concat([r, b], axis=1).dropna()
    if len(d) < 10:
        return np.nan, np.nan
    x, y = d.iloc[:, 1].to_numpy(), d.iloc[:, 0].to_numpy()
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = float((y.mean() - beta * x.mean()) * periods)
    return beta, alpha


def hit_rate(r: pd.Series, b: pd.Series, freq: str = "W-FRI") -> float:
    d = pd.concat([r, b], axis=1).dropna()
    w = (1 + d).resample(freq).prod() - 1
    return float((w.iloc[:, 0] > w.iloc[:, 1]).mean()) if len(w) else np.nan


def expected_max_sharpe(sr_var: float, n_trials: int) -> float:
    """Expected maximum (per-period) Sharpe of `n_trials` unskilled strategies."""
    if n_trials < 2 or sr_var <= 0:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                    + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe_ratio(r: pd.Series, trial_sharpes: list[float]) -> float:
    """Probability that the strategy's true Sharpe exceeds the best Sharpe expected from
    `len(trial_sharpes)` unskilled trials. `trial_sharpes` are *per-period* Sharpe ratios of all
    variants tried (including this one)."""
    r = r.dropna()
    T = len(r)
    if T < 3 or r.std() == 0:
        return np.nan
    sr = float(r.mean() / r.std(ddof=1))
    sr0 = expected_max_sharpe(float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0,
                              len(trial_sharpes))
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def summarize(daily: pd.DataFrame, bench: pd.Series | None, rf_annual: float) -> dict:
    """`daily` has net_ret, gross_ret, turnover, n_holdings (from the backtest engine)."""
    r = daily["net_ret"]
    out: dict[str, Any] = {
        "start": str(daily.index.min().date()), "end": str(daily.index.max().date()), "days": int(len(r)),
        "cagr": cagr(r), "ann_vol": ann_vol(r), "sharpe": sharpe(r, rf_annual), "sortino": sortino(r, rf_annual),
        "max_drawdown": max_drawdown(r), "calmar": calmar(r),
        "gross_cagr": cagr(daily["gross_ret"]),
    }
    out["cost_drag"] = out["gross_cagr"] - out["cagr"]
    if "turnover" in daily:
        out["annual_turnover"] = float(daily["turnover"].sum() * PERIODS / max(1, len(daily)))
    if "n_holdings" in daily:
        out["avg_holdings"] = float(daily["n_holdings"].mean())
    if bench is not None:
        b = bench.reindex(daily.index)
        out["beta"], out["alpha"] = beta_alpha(r, b)
        out["hit_rate_weekly"] = hit_rate(r, b)
    return out
