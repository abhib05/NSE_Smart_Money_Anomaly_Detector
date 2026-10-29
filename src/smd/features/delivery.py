"""Delivery features - the India-specific core of the project.

Delivery % = share of traded quantity actually settled into demat accounts (not squared
off intraday). Accumulation by long-horizon buyers should show up as unusually high
delivery, ideally while price stays contained ("quiet accumulation").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.features.base import FeatureContext, safe_log, trailing_z
from smd.features.price import realized_vol, ret_n, volume_z_60


def delivery_pct(ctx: FeatureContext) -> pd.DataFrame:
    return ctx["dpct"]


def delivery_pct_z_60(ctx: FeatureContext) -> pd.DataFrame:
    return ctx.memo("dpct_z", lambda: trailing_z(ctx["dpct"], 60, 40))


def delivery_qty_z_60(ctx: FeatureContext) -> pd.DataFrame:
    return ctx.memo("dqty_z", lambda: trailing_z(safe_log(ctx["dqty"]), 60, 40))


def delivery_volume_divergence(ctx: FeatureContext) -> pd.DataFrame:
    return delivery_qty_z_60(ctx) - volume_z_60(ctx)


def quiet_accumulation(ctx: FeatureContext) -> pd.DataFrame:
    """1 if delivery-% z > 2 while |5d return| < one 5-day volatility (21d daily vol * sqrt 5)."""
    z = delivery_pct_z_60(ctx)
    r5 = ret_n(5)(ctx)
    vol5 = realized_vol(21)(ctx) * np.sqrt(5)
    flag = ((z > 2) & (r5.abs() < vol5)).astype(float)
    return flag.where(z.notna() & r5.notna() & vol5.notna())
