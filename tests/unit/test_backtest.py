"""Backtest, portfolio, cost and metric tests (property-based where it matters)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from smd.backtest.costs import round_trip_cost_bps, trade_cost
from smd.backtest.engine import MarketData, ew_universe_returns, run_backtest
from smd.config import CostsCfg, SlippageCfg, load_config
from smd.metrics import performance as pf
from smd.portfolio.construct import apply_caps, hrp_weights, select_names, target_weights


# ---------------------------------------------------------------- costs
def test_costs_nonnegative_and_side_specific():
    cfg = CostsCfg()
    b = trade_cost(100_000, "buy", 1e8, cfg)
    s = trade_cost(100_000, "sell", 1e8, cfg)
    assert all(v >= 0 for v in [*b.values(), *s.values()])
    assert b["stamp"] > 0 and s["stamp"] == 0
    assert s["dp"] > 0 and b["dp"] == 0
    assert b["stt"] == pytest.approx(100.0)  # 0.1% STT on delivery buy
    rt = round_trip_cost_bps(100_000, 1e8, cfg)
    assert 25 < rt < 60  # ~20 bps STT + ~2 bps fees/stamp + slippage + DP


def test_slippage_grows_with_participation():
    cfg = CostsCfg()
    small = trade_cost(10_000, "buy", 1e8, cfg)["slippage"] / 10_000
    big = trade_cost(10_000, "buy", 1e5, cfg)["slippage"] / 10_000
    assert big > small


# ---------------------------------------------------------------- portfolio
def test_select_names_turnover_buffer():
    scores = pd.Series({f"S{i}": -i for i in range(20)})  # S0 best
    picked = select_names(scores, held={"S6"}, top_k=5, buffer=1.5)  # S6 rank 7 <= 7.5 -> kept
    assert "S6" in picked and len(picked) == 5
    picked = select_names(scores, held={"S9"}, top_k=5, buffer=1.5)  # rank 10 > 7.5 -> dropped
    assert "S9" not in picked


@settings(max_examples=50, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(st.floats(0.01, 1.0), min_size=2, max_size=15), st.floats(0.05, 0.5))
def test_apply_caps_respects_caps_and_budget(raw, cap):
    w = pd.Series(raw)
    w = w / w.sum()
    out = apply_caps(w, pd.Series(cap, index=w.index))
    assert (out <= cap + 1e-9).all()
    assert out.sum() <= 1 + 1e-9
    assert out.sum() == pytest.approx(min(1.0, cap * len(w)), rel=1e-6)


def test_liquidity_cap_leaves_cash():
    w = target_weights(["A", "B"], "equal", None, 0.6, pd.Series({"A": 0.1, "B": 0.1}))
    assert w.sum() == pytest.approx(0.2)


def test_hrp_weights_sum_to_one():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (120, 6)), columns=list("ABCDEF"))
    r["F"] *= 4  # riskier asset gets less weight
    w = hrp_weights(list("ABCDEF"), r)
    assert w.sum() == pytest.approx(1.0)
    assert w["F"] < w[["A", "B", "C", "D", "E"]].mean()


# ---------------------------------------------------------------- engine
def _market(n_days, n_stocks, seed, vol=0.01):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-01", periods=n_days)
    secs = [f"S{i}" for i in range(n_stocks)]
    rets = rng.normal(0.0003, vol, (n_days, n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=secs)
    opn = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, vol / 3, (n_days, n_stocks)))
    tv = pd.DataFrame(1e9, index=days, columns=secs)
    uni = pd.DataFrame(True, index=days, columns=secs)
    return MarketData(days, opn, close, tv, tv, uni)


def _cfg(top_k, costs=True, drift=0.25):
    cfg = load_config()
    cfg.portfolio.top_k = top_k
    cfg.portfolio.max_weight = 1.0
    cfg.portfolio.drift_tolerance = drift
    if not costs:
        cfg.costs = CostsCfg(stt_buy=0, stt_sell=0, exchange_txn=0, sebi_fee=0, stamp_duty_buy=0,
                             dp_charge_per_sell_inr=0, slippage=SlippageCfg(base_bps=0, impact_bps_at_1pct=0))
    return cfg


def _weekly_random_signals(md, seed):
    rng = np.random.default_rng(seed)
    fridays = [d for d in md.days if d.weekday() == 4]
    return {d: pd.Series(rng.normal(size=md.close_px.shape[1]), index=md.close_px.columns) for d in fridays}


@settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
@given(n_days=st.integers(25, 70), n_stocks=st.integers(2, 8), top_k=st.integers(1, 6), seed=st.integers(0, 10_000),
       missing=st.floats(0.0, 0.15))
def test_invariants_hold_on_random_markets(n_days, n_stocks, top_k, seed, missing):
    md = _market(n_days, n_stocks, seed)
    rng = np.random.default_rng(seed + 1)
    holes = rng.random(md.close_px.shape) < missing  # random missing prices (no trading)
    md.close_px = md.close_px.mask(holes)
    md.exec_px = md.exec_px.mask(holes)
    res = run_backtest(md, _weekly_random_signals(md, seed), _cfg(min(top_k, n_stocks)), "equal",
                       check_invariants=True)
    d = res.daily
    assert (d["costs"] >= 0).all()
    assert np.isfinite(d["equity"]).all() and (d["equity"] > 0).all()
    if len(res.trades):
        buys = res.trades[res.trades["side"] == "buy"]
        for r in buys.itertuples():
            assert np.isfinite(md.exec_px.loc[r.date, r.isin])  # never trades without a price


def test_zero_signal_matches_ew_benchmark_minus_costs():
    md = _market(250, 6, seed=3)
    sig = _weekly_random_signals(md, 3)
    free = run_backtest(md, sig, _cfg(6, costs=False, drift=0.0), "equal")
    costly = run_backtest(md, sig, _cfg(6, costs=True, drift=0.0), "equal")
    ew = ew_universe_returns(md).reindex(free.daily.index).fillna(0)
    start = free.daily.index[free.daily["n_holdings"] > 0][0]
    gap = abs(pf.cagr(free.daily["net_ret"][start:]) - pf.cagr(ew[start:]))
    assert gap < 0.02, f"zero-signal, zero-cost strategy deviates from EW by {gap:.4f}"
    assert pf.cagr(costly.daily["net_ret"]) < pf.cagr(free.daily["net_ret"])
    assert costly.daily["costs"].sum() > 0 and free.daily["costs"].sum() == 0


def test_delisted_holding_is_exited_with_haircut():
    md = _market(60, 3, seed=1)
    md.close_px.loc[md.days[30]:, "S0"] = np.nan
    md.exec_px.loc[md.days[30]:, "S0"] = np.nan
    sig = {d: pd.Series({"S0": 3.0, "S1": 2.0, "S2": 1.0}) for d in md.days[:20] if d.weekday() == 4}
    res = run_backtest(md, sig, _cfg(3), "equal")
    ev = [e for e in res.events if e["isin"] == "S0"]
    assert len(ev) == 1 and ev[0]["event"] == "delisted_exit"


# ---------------------------------------------------------------- metrics
def test_metrics_match_hand_computation():
    r = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    mean, sd = r.mean(), r.std(ddof=1)
    assert pf.sharpe(r, 0.0) == pytest.approx(mean / sd * np.sqrt(252))
    assert pf.max_drawdown(r) == pytest.approx(-0.02)
    growth = 1.01 * 0.98 * 1.03
    assert pf.cagr(r) == pytest.approx(growth ** (252 / 4) - 1)
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert pf.sortino(r, 0.0) == pytest.approx(mean / down * np.sqrt(252))


def test_drawdown_series():
    r = pd.Series([0.1, -0.5, 0.2])
    assert pf.drawdown(r).tolist() == pytest.approx([0.0, -0.5, -0.4])


def test_beta_alpha_recovers_known_values():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 2000))
    r = 0.0002 + 1.5 * b
    beta, alpha = pf.beta_alpha(r, b)
    assert beta == pytest.approx(1.5) and alpha == pytest.approx(0.0002 * 252)


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000))
    sr = r.mean() / r.std()
    few = pf.deflated_sharpe_ratio(r, [sr, sr * 0.5])
    many = pf.deflated_sharpe_ratio(r, list(np.linspace(-sr, sr, 50)))
    assert 0 <= many < few <= 1
