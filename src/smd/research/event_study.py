"""Event study around high accumulation scores.

Event: a stock's accumulation score is in the top `top_pct` of the universe on day *t*
(and > 0). Events of the same stock within `min_gap` days of a previous event are dropped
so windows don't double count. For each event we track daily excess returns (stock TR
return minus the equal-weight universe return) from day -20 to +40, and compare with
**matched controls**: non-event universe stocks on the same date in the same liquidity and
volatility terciles. Confidence bands cluster by event date (events on the same day are
not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def excess_return_matrix(panel: pd.DataFrame, universe: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    p = panel.loc[~panel["dq_exclude"], ["date", "isin", "adj_close_tr"]]
    px = p.pivot(index="date", columns="isin", values="adj_close_tr").reindex(days)
    ret = px / px.ffill(limit=5).shift(1) - 1
    u = universe[universe["in_universe"]]
    mask = pd.DataFrame(False, index=days, columns=px.columns)
    arr = mask.to_numpy()
    ci = px.columns.get_indexer(u["isin"])
    ok = ci >= 0
    arr[days.get_indexer(u["date"])[ok], ci[ok]] = True
    ew = ret.where(arr).mean(axis=1)
    return ret.sub(ew, axis=0)


def select_events(scores: pd.DataFrame, col: str, top_pct: float, min_gap: int, days: pd.DatetimeIndex) -> pd.DataFrame:
    s = scores[["date", "isin", col]].dropna()
    s = s[s[col] > 0]
    n_universe = scores.groupby("date")["isin"].transform("size").reindex(s.index)
    rank = s.groupby("date")[col].rank(ascending=False, method="first")
    ev = s[rank <= np.maximum(1, np.floor(top_pct * n_universe))].copy()
    ev["pos"] = days.get_indexer(ev["date"])
    ev = ev.sort_values(["isin", "pos"])
    keep: list = []
    last: dict[str, int] = {}
    for r in ev.itertuples():
        if r.isin not in last or r.pos - last[r.isin] > min_gap:
            keep.append(r.Index)
            last[r.isin] = r.pos
    return ev.loc[keep].sort_values("date").reset_index(drop=True)


def _terciles(x: pd.Series, by: pd.Series) -> pd.Series:
    return x.groupby(by).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else 0)


def event_study(events: pd.DataFrame, er: pd.DataFrame, controls_pool: pd.DataFrame, window: tuple[int, int]
                ) -> dict[str, pd.DataFrame]:
    """`controls_pool`: universe rows (date, isin, med_traded_value, vol_21d) for matching."""
    lo, hi = window
    rel = np.arange(lo, hi + 1)
    T = len(er)
    cols = er.columns
    ER = er.to_numpy()
    pool = controls_pool.dropna(subset=["med_traded_value", "vol_21d"]).copy()
    pool["liq_t"] = _terciles(pool["med_traded_value"], pool["date"])
    pool["vol_t"] = _terciles(pool["vol_21d"], pool["date"])
    pool = pool.set_index(["date", "isin"])
    ev_keys = set(zip(events["date"], events["isin"], strict=False))
    by_date = {d: g for d, g in pool.reset_index().groupby("date")}

    ev_paths, ctrl_paths, dates = [], [], []
    for e in events.itertuples():
        c = cols.get_loc(e.isin) if e.isin in cols else None
        if c is None or e.pos + lo < 0 or e.pos + hi >= T:
            continue
        rows = np.arange(e.pos + lo, e.pos + hi + 1)
        path = ER[rows, c]
        g = by_date.get(e.date)
        if g is None or (e.date, e.isin) not in pool.index:
            continue
        me = pool.loc[(e.date, e.isin)]
        m = g[(g["liq_t"] == me["liq_t"]) & (g["vol_t"] == me["vol_t"])]
        m = m[[(e.date, i) not in ev_keys for i in m["isin"]]]
        ci = cols.get_indexer(m["isin"])
        ci = ci[ci >= 0]
        if len(ci) == 0:
            continue
        ctrl = np.nanmean(ER[np.ix_(rows, ci)], axis=1)
        ev_paths.append(np.nan_to_num(path))
        ctrl_paths.append(np.nan_to_num(ctrl))
        dates.append(e.date)
    if not ev_paths:
        return {"car": pd.DataFrame(), "n_events": pd.DataFrame({"n": [0]})}
    E, C = np.array(ev_paths), np.array(ctrl_paths)
    D = E - C
    dd = pd.Series(dates)
    out = {}
    for name, M in (("event", E), ("control", C), ("diff", D)):
        car = np.cumsum(M, axis=1)
        # cluster by event date: average paths of same-day events, then SE across dates
        per_date = pd.DataFrame(car).groupby(dd.values).mean()
        mean = per_date.mean().to_numpy()
        se = per_date.std(ddof=1).to_numpy() / np.sqrt(max(1, len(per_date)))
        out[name] = pd.DataFrame({"rel_day": rel, "car": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se})
    car = out["event"][["rel_day"]].copy()
    for name in ("event", "control", "diff"):
        car[f"{name}_car"] = out[name]["car"]
        car[f"{name}_lo"] = out[name]["lo"]
        car[f"{name}_hi"] = out[name]["hi"]
    # post-event (tradeable) CAR from day +1 to +h, difference vs controls
    post = {}
    zero = int(np.where(rel == 0)[0][0])
    for h in (5, 20, 40):
        if zero + h < D.shape[1]:
            v = pd.Series(D[:, zero + 1: zero + h + 1].sum(axis=1)).groupby(dd.values).mean()
            post[h] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(max(1, len(v)))),
                       "n_dates": int(len(v))}
    return {"car": car, "n_events": pd.DataFrame({"n": [len(E)], "n_dates": [dd.nunique()]}),
            "post": pd.DataFrame(post).T.rename_axis("horizon").reset_index()}
