"""Price, volatility and liquidity features (all trailing, data <= t)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.features.base import FeatureContext, rolling_beta, safe_log, trailing_z


def _traded_today(ctx: FeatureContext, x: pd.DataFrame) -> pd.DataFrame:
    return x.where(ctx["c"].notna())


def ret_n(n: int):
    def f(ctx: FeatureContext) -> pd.DataFrame:
        return _traded_today(ctx, ctx.c_ff / ctx.c_ff.shift(n) - 1.0)

    return f


def mom_126_5(ctx: FeatureContext) -> pd.DataFrame:
    return _traded_today(ctx, ctx.c_ff.shift(5) / ctx.c_ff.shift(126) - 1.0)


def realized_vol(n: int):
    def f(ctx: FeatureContext) -> pd.DataFrame:
        return ctx.ret1.rolling(n, min_periods=int(n * 0.7)).std()

    return f


def downside_vol_63(ctx: FeatureContext) -> pd.DataFrame:
    neg = ctx.ret1.clip(upper=0.0)
    return np.sqrt((neg**2).rolling(63, min_periods=44).mean())


def dist_52w_high(ctx: FeatureContext) -> pd.DataFrame:
    hi = ctx["c"].rolling(252, min_periods=126).max()
    return ctx["c"] / hi - 1.0


def clv(ctx: FeatureContext) -> pd.DataFrame:
    """Close location value within the day's range: +1 = closed at high, -1 = at low."""
    hi, lo, c = ctx["h"], ctx["l"], ctx["c"]
    rng = (hi - lo).where((hi - lo) > 0)
    return ((c - lo) - (hi - c)) / rng


def clv_5d(ctx: FeatureContext) -> pd.DataFrame:
    return clv(ctx).rolling(5, min_periods=3).mean()


def beta_126(ctx: FeatureContext) -> pd.DataFrame:
    return rolling_beta(ctx.ret1, ctx.mkt_ret, 126, 80)


def log_med_tv_21(ctx: FeatureContext) -> pd.DataFrame:
    return safe_log(ctx["tv"].rolling(21, min_periods=15).median())


def med_tv_21(ctx: FeatureContext) -> pd.DataFrame:
    return ctx.memo("med_tv_21", lambda: ctx["tv"].rolling(21, min_periods=15).median())


def volume_z_60(ctx: FeatureContext) -> pd.DataFrame:
    return ctx.memo("vol_z_60", lambda: trailing_z(safe_log(ctx["v"]), 60, 40))


def amihud_21(ctx: FeatureContext) -> pd.DataFrame:
    """Mean |return| per ₹ crore traded (Amihud illiquidity)."""
    illiq = ctx.ret1.abs() / (ctx["tv"].where(ctx["tv"] > 0) / 1e7)
    return illiq.rolling(21, min_periods=15).mean()


def range_pct(ctx: FeatureContext) -> pd.DataFrame:
    return (ctx["h"] - ctx["l"]) / ctx["c"]
