"""Backtest stage: every ranker variant x weighting scheme, plus benchmarks and a
random-signal sanity strategy. All trials are reported (and counted for the deflated
Sharpe ratio), not just the best."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smd.backtest.engine import ew_universe_returns, market_data_from_panel, run_backtest  # noqa: E402
from smd.config import Config  # noqa: E402
from smd.metrics.performance import deflated_sharpe_ratio, drawdown, summarize  # noqa: E402
from smd.model.ranker import VARIANTS  # noqa: E402
from smd.research.run import period_bounds  # noqa: E402
from smd.utils import get_logger, read_dataset  # noqa: E402

log = get_logger(__name__)
WEIGHTINGS = ("equal", "hrp")


def _signals(preds: pd.DataFrame, variant: str) -> dict[pd.Timestamp, pd.Series]:
    g = preds[preds["model_variant"] == variant]
    return {d: x.set_index("isin")["score"] for d, x in g.groupby("date")}


def _random_signals(preds: pd.DataFrame, seed: int) -> dict[pd.Timestamp, pd.Series]:
    rng = np.random.default_rng(seed)
    g = preds[preds["model_variant"] == VARIANTS[0]]
    return {d: pd.Series(rng.normal(size=len(x)), index=x["isin"].to_numpy()) for d, x in g.groupby("date")}


def run_backtests(cfg: Config, run, period: str = "dev", variants=VARIANTS, weightings=WEIGHTINGS) -> dict:
    from smd.clean.panel import trading_days

    lo, hi = period_bounds(cfg, period)
    out_dir = run.dir / f"backtest_{period}"
    out_dir.mkdir(parents=True, exist_ok=True)
    days = trading_days(cfg)
    panel = read_dataset(cfg, "silver", "daily_panel",
                         columns=["date", "isin", "adj_open_tr", "adj_close_tr", "traded_value", "dq_exclude"])
    uni = read_dataset(cfg, "gold", "universe")
    md = market_data_from_panel(panel, uni, days, cfg.labels.execution)
    preds = read_dataset(cfg, "gold", "predictions")
    preds = preds[(preds["date"] >= lo) & (preds["date"] <= hi)]
    if preds.empty:
        raise RuntimeError(f"no predictions in the {period} period")
    start = preds["date"].min()
    end = min(days[-1], hi)

    bench = read_dataset(cfg, "silver", "benchmark").set_index("date")["nifty500"].reindex(days)
    nifty = (bench / bench.shift(1) - 1)
    ew = ew_universe_returns(md)

    results, metrics = {}, []
    strategies = [(v, w) for v in variants for w in weightings] + [("random_signal", "equal")]
    for v, w in strategies:
        sig = _random_signals(preds, cfg.seed) if v == "random_signal" else _signals(preds, v)
        name = f"{v}|{w}"
        res = run_backtest(md, sig, cfg, w, start, end)
        results[name] = res
        m = summarize(res.daily, nifty, cfg.backtest.rf_annual)
        m.update(strategy=name, variant=v, weighting=w, n_delisting_exits=len(res.events))
        metrics.append(m)
        res.daily.to_csv(out_dir / f"daily_{v.replace('+', '_')}_{w}.csv")
        log.info("%s: CAGR %.2f%% Sharpe %.2f MDD %.1f%%", name, 100 * m["cagr"], m["sharpe"], 100 * m["max_drawdown"])

    # benchmarks over the same window
    idx = results[next(iter(results))].daily.index
    bench_rets = {"Nifty 500 (price index)": nifty.reindex(idx).fillna(0), "EW universe (TR, no costs)": ew.reindex(idx).fillna(0)}
    for name, r in bench_rets.items():
        m = summarize(pd.DataFrame({"net_ret": r, "gross_ret": r}), nifty, cfg.backtest.rf_annual)
        m.update(strategy=name, variant="benchmark", weighting="-")
        metrics.append(m)

    # deflated Sharpe: all non-sanity trials count
    trials = [r for k, r in results.items() if not k.startswith("random_signal")]
    trial_sr = [float(t.daily["net_ret"].mean() / t.daily["net_ret"].std()) for t in trials]
    for m in metrics:
        if m["variant"] not in ("benchmark", "random_signal"):
            m["deflated_sharpe"] = deflated_sharpe_ratio(results[m["strategy"]].daily["net_ret"], trial_sr)
            m["n_trials"] = len(trials)
    table = pd.DataFrame(metrics)
    lead = ["strategy", "cagr", "gross_cagr", "cost_drag", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
            "annual_turnover", "avg_holdings", "hit_rate_weekly", "beta", "alpha", "deflated_sharpe", "n_trials"]
    table = table[[c for c in lead if c in table] + [c for c in table if c not in lead]]
    table.to_csv(out_dir / "metrics.csv", index=False)
    costs = pd.DataFrame({k: r.cost_breakdown for k, r in results.items()}).T.fillna(0.0)
    costs.to_csv(out_dir / "cost_breakdown.csv")
    pd.concat([r.positions.assign(strategy=k) for k, r in results.items() if len(r.positions)]).to_csv(
        out_dir / "positions.csv", index=False)
    _plots(out_dir, results, bench_rets)
    (out_dir / "events.json").write_text(json.dumps({k: r.events for k, r in results.items()}, default=str, indent=1),
                                         encoding="utf-8")
    return {"period": period, "sharpe": table.set_index("strategy")["sharpe"].round(3).to_dict()}


def _plots(out_dir, results, bench_rets) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, res in results.items():
        eq = (1 + res.daily["net_ret"]).cumprod()
        style = ":" if name.startswith("random") else ("-" if name.endswith("equal") else "--")
        axes[0].plot(eq.index, eq, style, lw=1.2, label=name)
        axes[1].plot(eq.index, drawdown(res.daily["net_ret"]) * 100, style, lw=0.9)
    for name, r in bench_rets.items():
        eq = (1 + r).cumprod()
        axes[0].plot(eq.index, eq, color="k" if "Nifty" in name else "grey", lw=2, label=name)
        axes[1].plot(eq.index, drawdown(r) * 100, color="k" if "Nifty" in name else "grey", lw=1.5)
    axes[0].set_ylabel("growth of 1 (net of costs)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("drawdown (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=120)
    plt.close(fig)
