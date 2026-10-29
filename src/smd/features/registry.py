"""Feature registry: every feature has a name, function, lookback, group and description.

Groups drive the ranking-model ablation: `price` + `liquidity` = base, `delivery` + `bulk` =
the India-specific smart-money features. `behaviour` channels (prefixed `bh_`) are the
inputs of the anomaly models, not ranker features.

`build_features()` returns two long tables:
* `features`  - one row per (date, isin) *in the universe*, all registered features;
* `behaviour` - one row per (date, isin) for every security that is ever in the universe
  (anomaly models need the 30-day history before a stock enters the universe).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smd.config import Config
from smd.features import bulk_deals as bd
from smd.features import delivery as dl
from smd.features import price as px
from smd.features.base import FeatureContext, build_context, trailing_z


@dataclass(frozen=True)
class Feature:
    name: str
    func: Callable[[FeatureContext], pd.DataFrame]
    lookback: int
    group: str
    description: str


def _registry(cfg: Config | None = None) -> list[Feature]:
    cap = cfg.features.bulk_days_since_cap if cfg else 60
    return [
        Feature("ret_5d", px.ret_n(5), 5, "price", "5-day price return"),
        Feature("ret_21d", px.ret_n(21), 21, "price", "21-day price return"),
        Feature("ret_63d", px.ret_n(63), 63, "price", "63-day price return"),
        Feature("ret_126d", px.ret_n(126), 126, "price", "126-day price return"),
        Feature("mom_126_5", px.mom_126_5, 126, "price", "126-day momentum skipping the last 5 days"),
        Feature("vol_21d", px.realized_vol(21), 21, "price", "21-day realised daily volatility"),
        Feature("vol_63d", px.realized_vol(63), 63, "price", "63-day realised daily volatility"),
        Feature("downvol_63d", px.downside_vol_63, 63, "price", "63-day downside semi-deviation"),
        Feature("dist_52w_high", px.dist_52w_high, 252, "price", "close / 252-day max close - 1"),
        Feature("clv_5d", px.clv_5d, 5, "price", "5-day mean close location within the day's range"),
        Feature("beta_126d", px.beta_126, 126, "price", "126-day beta vs equal-weight universe"),
        Feature("log_med_tv_21d", px.log_med_tv_21, 21, "liquidity", "log 21-day median traded value"),
        Feature("volume_z_60d", px.volume_z_60, 60, "liquidity", "log volume z vs previous 60 days"),
        Feature("amihud_21d", px.amihud_21, 21, "liquidity", "Amihud illiquidity, 21 days"),
        Feature("delivery_pct", dl.delivery_pct, 1, "delivery", "delivery % of traded quantity"),
        Feature("delivery_pct_z_60d", dl.delivery_pct_z_60, 60, "delivery", "delivery % z vs own previous 60 days"),
        Feature("delivery_qty_z_60d", dl.delivery_qty_z_60, 60, "delivery", "log delivered quantity z vs previous 60 days"),
        Feature("delivery_volume_div", dl.delivery_volume_divergence, 60, "delivery", "delivery qty z - volume z"),
        Feature("quiet_accumulation", dl.quiet_accumulation, 60, "delivery", "delivery z > 2 with |5d ret| < 5d vol"),
        Feature("bulk_net_buy_5d", bd.net_buy_n(5), 21, "bulk", "net bulk/block buy value 5d / 21d median traded value"),
        Feature("bulk_net_buy_20d", bd.net_buy_n(20), 21, "bulk", "net bulk/block buy value 20d / 21d median traded value"),
        Feature("bulk_buyers_20d", bd.distinct_buyers_20, 20, "bulk", "distinct buying entities over 20 days"),
        Feature("bulk_imbalance_20d", bd.imbalance_20, 20, "bulk", "(buy - sell) / (buy + sell) deal value, 20 days"),
        Feature("bulk_days_since_buy", bd.days_since_bulk_buy(cap), cap, "bulk", f"trading days since last bulk buy (cap {cap})"),
    ]


REGISTRY: dict[str, Feature] = {f.name: f for f in _registry()}
# cross-sectional features computed on the universe after the per-security ones
XS_FEATURES = {"delivery_z_xs_rank": ("delivery_pct_z_60d", "delivery")}

BEHAVIOUR_CHANNELS = [
    "bh_delivery_pct_z", "bh_delivery_qty_z", "bh_volume_z", "bh_ret_z", "bh_range_z",
    "bh_clv", "bh_bulk_net", "bh_turnover_rank",
]


def feature_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for f in REGISTRY.values():
        groups.setdefault(f.group, []).append(f.name)
    for name, (_, g) in XS_FEATURES.items():
        groups.setdefault(g, []).append(name)
    return groups


def behaviour_channels(ctx: FeatureContext, turnover_rank: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compact behaviour vector for the anomaly models; each channel standardised with past data only.
    `turnover_rank` must be ranked across *all* securities traded that day (a point-in-time set)."""
    return {
        "bh_delivery_pct_z": dl.delivery_pct_z_60(ctx),
        "bh_delivery_qty_z": dl.delivery_qty_z_60(ctx),
        "bh_volume_z": px.volume_z_60(ctx),
        "bh_ret_z": trailing_z(ctx.ret1, 252, 60),
        "bh_range_z": trailing_z(px.range_pct(ctx), 252, 60),
        "bh_clv": px.clv(ctx),
        "bh_bulk_net": bd.net_buy_1d(ctx).clip(-5, 5),
        "bh_turnover_rank": turnover_rank,
    }


def _stack(frames: dict[str, pd.DataFrame], mask: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = {}
    for name, w in frames.items():
        w = w.replace([np.inf, -np.inf], np.nan)
        if mask is not None:
            w = w.where(mask)
        cols[name] = w.stack(future_stack=True)
    out = pd.DataFrame(cols)
    out.index.names = ["date", "isin"]
    return out


def build_features(panel: pd.DataFrame, deals: pd.DataFrame, universe: pd.DataFrame,
                   days: pd.DatetimeIndex, cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ever = sorted(universe.loc[universe["in_universe"], "isin"].unique())
    ctx = build_context(panel, deals, universe, days, ever)
    feats = {f.name: f.func(ctx) for f in _registry(cfg)}
    features = _stack(feats, ctx.universe_mask).dropna(how="all").reset_index()
    features = features.merge(
        universe.loc[universe["in_universe"], ["date", "isin"]], on=["date", "isin"], how="right"
    )
    for name, (src, _) in XS_FEATURES.items():
        features[name] = features.groupby("date")[src].rank(pct=True) * 2 - 1
    full = panel[["date", "isin", "traded_value", "dq_exclude"]]
    full = full[~full["dq_exclude"]]
    rank = full.groupby("date")["traded_value"].rank(pct=True) * 2 - 1
    turnover_rank = (
        full.assign(r=rank).pivot(index="date", columns="isin", values="r")
        .reindex(index=days, columns=ctx["c"].columns)
    )
    beh = behaviour_channels(ctx, turnover_rank)
    behaviour = _stack(beh).reset_index()
    behaviour = behaviour[ctx["c"].stack(future_stack=True).notna().to_numpy()]
    return (features.sort_values(["date", "isin"]).reset_index(drop=True),
            behaviour.sort_values(["isin", "date"]).reset_index(drop=True))


def normalize_cross_section(df: pd.DataFrame, cols: list[str], winsor: float = 0.01) -> pd.DataFrame:
    """Per date: winsorise at the `winsor`/1-`winsor` quantiles, then rank to [-1, 1]. NaN -> 0 (neutral)."""
    out = df[["date", *cols]].copy()
    g = out.groupby("date")
    for c in cols:
        lo = g[c].transform(lambda s: s.quantile(winsor))
        hi = g[c].transform(lambda s: s.quantile(1 - winsor))
        clipped = out[c].clip(lo, hi)
        k = clipped.groupby(out["date"]).rank(method="average")  # 1..n
        n = clipped.groupby(out["date"]).transform("count")
        out[c] = ((k - 1) / (n - 1).where(n > 1) * 2 - 1).fillna(0.0)
    return out[cols]
