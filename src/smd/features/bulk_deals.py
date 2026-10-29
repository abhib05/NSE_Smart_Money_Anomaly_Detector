"""Bulk/block deal features. Deals are disclosed by NSE on the trade date after close,
so a deal dated *t* is usable at the close of *t*."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.features.base import FeatureContext
from smd.features.price import med_tv_21


def _daily_values(ctx: FeatureContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    def build():
        d = ctx.deals
        cols = ctx["c"].columns
        buy = pd.DataFrame(0.0, index=ctx.days, columns=cols)
        sell = buy.copy()
        if len(d):
            d = d[d["date"].isin(ctx.days)]
            g = d.groupby(["date", "isin", "side"])["value"].sum().unstack("side", fill_value=0.0)
            for side, target in (("BUY", buy), ("SELL", sell)):
                if side in g:
                    s = g[side].unstack("isin").reindex(index=ctx.days, columns=cols).fillna(0.0)
                    target += s
        return pd.concat({"BUY": buy, "SELL": sell}, axis=1)

    both = ctx.memo("deal_values", build)
    return both["BUY"], both["SELL"]


def _norm(ctx: FeatureContext, x: pd.DataFrame) -> pd.DataFrame:
    return x / med_tv_21(ctx).where(lambda s: s > 0)


def net_buy_n(n: int):
    def f(ctx: FeatureContext) -> pd.DataFrame:
        buy, sell = _daily_values(ctx)
        return _norm(ctx, (buy - sell).rolling(n, min_periods=1).sum())

    return f


def net_buy_1d(ctx: FeatureContext) -> pd.DataFrame:
    buy, sell = _daily_values(ctx)
    return _norm(ctx, buy - sell)


def imbalance_20(ctx: FeatureContext) -> pd.DataFrame:
    buy, sell = _daily_values(ctx)
    b, s = buy.rolling(20, min_periods=1).sum(), sell.rolling(20, min_periods=1).sum()
    tot = b + s
    return ((b - s) / tot.where(tot > 0)).fillna(0.0).where(ctx["c"].notna())


def distinct_buyers_20(ctx: FeatureContext) -> pd.DataFrame:
    """Number of distinct BUY-side clients over the last 20 trading days (inclusive of t)."""
    cols = ctx["c"].columns
    n_days = len(ctx.days)
    counts = np.zeros((n_days + 21, len(cols)))
    d = ctx.deals
    d = d[(d["side"] == "BUY") & d["date"].isin(ctx.days)]
    if len(d):
        pos = ctx.days.get_indexer(d["date"])
        cpos = cols.get_indexer(d["isin"])
        frame = pd.DataFrame({"pos": pos, "c": cpos, "client": d["client_name"].fillna("?").to_numpy()})
        for (c_any, _), g in frame.groupby(["c", "client"]):
            c = int(c_any)
            # merge overlapping [pos, pos+19] intervals so a client is counted once per day
            positions = sorted(int(x) for x in set(g["pos"]))
            start, end = positions[0], positions[0] + 19
            for p in positions[1:]:
                if p <= end + 1:
                    end = p + 19
                else:
                    counts[start, c] += 1
                    counts[end + 1, c] -= 1
                    start, end = p, p + 19
            counts[start, c] += 1
            counts[end + 1, c] -= 1
    out = pd.DataFrame(np.cumsum(counts, axis=0)[:n_days], index=ctx.days, columns=cols)
    return out.where(ctx["c"].notna())


def days_since_bulk_buy(cap: int):
    def f(ctx: FeatureContext) -> pd.DataFrame:
        buy, _ = _daily_values(ctx)
        pos = pd.DataFrame(np.arange(len(ctx.days))[:, None].repeat(buy.shape[1], axis=1),
                           index=buy.index, columns=buy.columns, dtype=float)
        last = pos.where(buy > 0).ffill()
        return (pos - last).fillna(cap).clip(upper=cap).where(ctx["c"].notna())

    return f
