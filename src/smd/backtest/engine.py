"""Daily backtest with weekly rebalancing.

Timeline for a rebalance date *t* (signal known at the close of *t*):
  close t   : select names + target weights (universe and liquidity as of t)
  open t+1  : execute (sells first, then buys) at the total-return-adjusted open price;
              each order is capped at `max_adv_participation` x that day's traded value
  close t+1 : mark to market at the adjusted close
Held names that keep their rank inside the buffer are only resized when their weight
drifts more than `drift_tolerance` (relative) from target.

A held security without a price is carried at its last close; after
`stale_price_exit_days` consecutive missing days it is treated as delisted and sold at
the last price minus `delisting_haircut` (logged).

Accounting invariants are checked every day (raise `AssertionError` when violated):
equity == cash + sum(holdings), cash >= 0, sum of weights <= 1, costs >= 0, no buy outside
the signal-date universe, no trade in a security without a price that day.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smd.backtest.costs import trade_cost
from smd.config import Config
from smd.portfolio.construct import select_names, target_weights

EPS = 1e-6


@dataclass
class MarketData:
    days: pd.DatetimeIndex
    exec_px: pd.DataFrame       # execution price (adj open TR for next_open)
    close_px: pd.DataFrame      # adj close TR (mark-to-market)
    traded_value: pd.DataFrame  # raw ₹ traded value
    med_tv21: pd.DataFrame
    in_universe: pd.DataFrame   # bool


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    events: list[dict] = field(default_factory=list)
    cost_breakdown: dict[str, float] = field(default_factory=dict)


def market_data_from_panel(panel: pd.DataFrame, universe: pd.DataFrame, days: pd.DatetimeIndex,
                           execution: str = "next_open") -> MarketData:
    p = panel[~panel["dq_exclude"]]
    u = universe[universe["in_universe"]]
    secs = sorted(u["isin"].unique())

    def wide(col: str) -> pd.DataFrame:
        return p.pivot(index="date", columns="isin", values=col).reindex(index=days, columns=secs)

    close = wide("adj_close_tr")
    exec_px = wide("adj_open_tr") if execution == "next_open" else close
    tv = wide("traded_value")
    mask = pd.DataFrame(False, index=days, columns=secs)
    arr = mask.to_numpy()
    arr[days.get_indexer(u["date"]), mask.columns.get_indexer(u["isin"])] = True
    return MarketData(days, exec_px, close, tv, tv.rolling(21, min_periods=15).median(),
                      pd.DataFrame(arr, index=days, columns=secs))


def run_backtest(md: MarketData, signals: Mapping[pd.Timestamp, pd.Series], cfg: Config, weighting: str = "equal",
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 check_invariants: bool = True) -> BacktestResult:
    pc, bc, cc = cfg.portfolio, cfg.backtest, cfg.costs
    drift_tol = pc.drift_tolerance
    days = md.days
    i0 = days.searchsorted(start) if start is not None else 0
    i1 = days.searchsorted(end, side="right") - 1 if end is not None else len(days) - 1
    cols = md.close_px.columns
    col_idx = {c: k for k, c in enumerate(cols)}
    C, X, TV = md.close_px.to_numpy(), md.exec_px.to_numpy(), md.traded_value.to_numpy()
    MED, U = md.med_tv21.to_numpy(), md.in_universe.to_numpy()

    cash = float(pc.notional_inr)
    units: dict[str, float] = {}
    last_px: dict[str, float] = {}
    stale: dict[str, int] = {}
    pending: pd.Series | None = None
    pending_universe: set[str] = set()
    rows, pos_rows, trade_rows, events = [], [], [], []
    cost_tot: dict[str, float] = {}
    prev_equity = cash

    for i in range(i0, i1 + 1):
        d = days[i]
        day_costs = 0.0
        traded = 0.0
        # ---- 1. execute pending orders at today's execution price
        if pending is not None:
            px_now = {s: X[i, col_idx[s]] for s in set(units) | set(pending.index)}
            equity_exec = cash + sum(u * (px_now[s] if np.isfinite(px_now[s]) else last_px[s]) for s, u in units.items())
            orders = []
            for s in set(units) | set(pending.index):
                p = px_now[s]
                if not np.isfinite(p) or p <= 0:
                    continue  # cannot trade without a price
                cur_val = units.get(s, 0.0) * p
                tgt_w = float(pending.get(s, 0.0))
                cur_w = cur_val / equity_exec if equity_exec > 0 else 0.0
                if s in units and tgt_w > 0 and abs(cur_w - tgt_w) <= drift_tol * tgt_w:
                    continue  # kept name within drift tolerance: no trade
                delta = tgt_w * equity_exec - cur_val
                cap = pc.max_adv_participation * TV[i, col_idx[s]] if np.isfinite(TV[i, col_idx[s]]) else 0.0
                delta = float(np.sign(delta) * min(abs(delta), cap))  # liquidity-capped fill
                full_exit = tgt_w == 0 and abs(delta) >= cur_val - EPS
                if abs(delta) >= pc.min_trade_inr or (full_exit and abs(delta) > EPS):
                    orders.append((s, delta, p))
            for s, delta, p in sorted(orders, key=lambda o: o[1]):  # sells first
                side = "buy" if delta > 0 else "sell"
                if side == "buy":
                    if check_invariants:
                        assert s in pending_universe, f"buy outside universe: {s} on {d}"
                    est = sum(trade_cost(delta, "buy", TV[i, col_idx[s]], cc).values())
                    if delta + est > cash:
                        delta = max(0.0, cash / (1 + est / delta) - EPS)
                    if delta <= EPS:
                        continue
                comp = trade_cost(delta, side, TV[i, col_idx[s]], cc)
                cost = sum(comp.values())
                if side == "sell" and cash - delta - cost < 0:
                    continue  # proceeds would not cover fixed charges
                for k, v in comp.items():
                    cost_tot[k] = cost_tot.get(k, 0.0) + v
                cash -= delta + cost
                units[s] = units.get(s, 0.0) + delta / p
                if units[s] * p < EPS:
                    units.pop(s)
                day_costs += cost
                traded += abs(delta)
                trade_rows.append({"date": d, "isin": s, "side": side, "value": delta, "price": p, "cost": cost})
            pending = None

        # ---- 2. mark to market; stale/delisted handling
        hold_val = 0.0
        for s in list(units):
            p = C[i, col_idx[s]]
            if np.isfinite(p) and p > 0:
                last_px[s] = p
                stale[s] = 0
            else:
                stale[s] = stale.get(s, 0) + 1
                if stale[s] > bc.stale_price_exit_days:
                    val = units[s] * last_px[s] * (1 - bc.delisting_haircut)
                    comp = trade_cost(val, "sell", 0.0, cc)
                    cost = min(val, sum(comp.values()))
                    cash += val - cost
                    day_costs += cost
                    events.append({"date": d, "isin": s, "event": "delisted_exit", "value": val,
                                   "haircut": bc.delisting_haircut})
                    units.pop(s)
                    continue
            hold_val += units[s] * last_px[s]
        equity = cash + hold_val

        if check_invariants:
            recomputed = cash + sum(u * last_px[s] for s, u in units.items())
            assert abs(recomputed - equity) <= 1e-6 * max(1.0, abs(equity)), "equity != cash + holdings"
            assert cash >= -1e-6 * max(1.0, equity), f"negative cash on {d}: {cash}"
            assert day_costs >= 0
            if equity > 0:
                assert hold_val / equity <= 1 + 1e-9, "weights sum > 1"

        net_ret = equity / prev_equity - 1 if prev_equity > 0 else 0.0
        rows.append({"date": d, "equity": equity, "cash": cash, "net_ret": net_ret,
                     "gross_ret": (equity + day_costs) / prev_equity - 1 if prev_equity > 0 else 0.0,
                     "costs": day_costs, "turnover": traded / 2 / prev_equity if prev_equity > 0 else 0.0,
                     "n_holdings": len(units)})
        prev_equity = equity

        # ---- 3. rebalance decision at the close of a signal date
        if d in signals and i < i1:
            sc = signals[d]
            uni_now = {s for s in sc.index if s in col_idx and U[i, col_idx[s]] and np.isfinite(C[i, col_idx[s]])}
            sc = sc[sc.index.isin(list(uni_now))]
            names = select_names(sc, set(units), pc.top_k, pc.sell_rank_buffer)
            rets = None
            if weighting == "hrp":
                lo = max(0, i - pc.hrp_lookback_days)
                win = md.close_px.iloc[lo: i + 1][names]
                rets = win / win.ffill().shift(1) - 1
            liq = pd.Series({s: pc.max_adv_participation * MED[i, col_idx[s]] / equity if equity > 0 else 0.0
                             for s in names})
            pending = target_weights(names, weighting, rets, pc.max_weight, liq.fillna(0.0))
            pending_universe = uni_now
            pos_rows += [{"rebalance_date": d, "isin": s, "weight": float(w)} for s, w in pending.items()]

    daily = pd.DataFrame(rows).set_index("date")
    return BacktestResult(daily, pd.DataFrame(pos_rows), pd.DataFrame(trade_rows), events, cost_tot)


def ew_universe_returns(md: MarketData) -> pd.Series:
    """Daily return of an equal-weight portfolio of yesterday's universe (no costs)."""
    r = md.close_px / md.close_px.shift(1) - 1
    return r.where(md.in_universe.shift(1, fill_value=False)).mean(axis=1)
