"""Signal research stage: answers the research question *before* any trading model.

For every anomaly model (LSTM-AE, Isolation Forest, delivery-z rule) and both the raw
anomaly percentile and the accumulation-filtered score:
1. rank IC vs fwd_excess_5d / fwd_excess_20d with block-bootstrap CIs,
2. quintile spreads,
3. event study (top `event_top_pct` accumulation scores, matched controls),
4. robustness by year and liquidity tercile.

Outputs go to `runs/<run_id>/research_<period>/` (CSV tables, PNG charts, summary JSON).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smd.config import Config  # noqa: E402
from smd.research.event_study import event_study, excess_return_matrix, select_events  # noqa: E402
from smd.research.ic_analysis import daily_rank_ic, ic_summary, quintile_returns  # noqa: E402
from smd.utils import get_logger, read_dataset  # noqa: E402

log = get_logger(__name__)

SCORE_TYPES = ("score", "accumulation_score")


def period_bounds(cfg: Config, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if period == "dev":
        return pd.Timestamp(cfg.study_start), pd.Timestamp(cfg.dev_end)
    if period == "test":
        return pd.Timestamp(cfg.test_start), pd.Timestamp.max
    raise ValueError(period)


def load_research_frame(cfg: Config, period: str) -> pd.DataFrame:
    lo, hi = period_bounds(cfg, period)
    scores = read_dataset(cfg, "gold", "anomaly_scores")
    scores = scores[(scores["date"] >= lo) & (scores["date"] <= hi)]
    wide = scores.pivot_table(index=["date", "isin"], columns="model", values=list(SCORE_TYPES))
    wide.columns = [f"{m}__{t}" for t, m in wide.columns]
    labels = read_dataset(cfg, "gold", "labels")
    feats = read_dataset(cfg, "gold", "features", columns=["date", "isin", "vol_21d", "log_med_tv_21d"])
    uni = read_dataset(cfg, "gold", "universe", columns=["date", "isin", "med_traded_value", "in_universe"])
    df = wide.reset_index().merge(labels, on=["date", "isin"], how="left")
    df = df.merge(feats, on=["date", "isin"], how="left").merge(uni.drop(columns="in_universe"), on=["date", "isin"], how="left")
    return df


def run_research(cfg: Config, run, period: str = "dev") -> dict:
    from smd.clean.panel import trading_days

    out_dir = run.dir / f"research_{period}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_research_frame(cfg, period)
    if df.empty:
        raise RuntimeError(f"no anomaly scores in the {period} period")
    score_cols = [c for c in df.columns if "__" in c]
    horizons = cfg.labels.horizons
    nb = cfg.research.bootstrap_samples

    # 1. IC table
    rows, ic_series = [], {}
    for sc in score_cols:
        for h in horizons:
            ic = daily_rank_ic(df, sc, f"fwd_excess_{h}d")
            ic_series[(sc, h)] = ic
            rows.append({"signal": sc, "horizon": h, **ic_summary(ic, h, nb, cfg.seed)})
    ic_table = pd.DataFrame(rows)
    ic_table.to_csv(out_dir / "ic_table.csv", index=False)

    # 2. quintiles (5-day horizon)
    qrows = []
    for sc in score_cols:
        tab, spread = quintile_returns(df, sc, f"fwd_excess_{horizons[0]}d")
        means = tab.mean()
        qrows.append({"signal": sc, **{f"Q{int(k)}": float(v) for k, v in means.items()},
                      "spread_mean": float(spread.mean()) if len(spread) else np.nan,
                      "spread_t": float(spread.mean() / spread.std() * np.sqrt(len(spread) / horizons[0]))
                      if len(spread) > 2 and spread.std() > 0 else np.nan})
    quint = pd.DataFrame(qrows)
    quint.to_csv(out_dir / "quintiles.csv", index=False)

    # 3. robustness: by year and liquidity tercile (5-day IC)
    df["year"] = df["date"].dt.year
    df["liq_tercile"] = df.groupby("date")["med_traded_value"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]) if len(s) >= 3 else "mid"
    )
    rob = []
    for sc in score_cols:
        for key in ("year", "liq_tercile"):
            for val, g in df.groupby(key, observed=True):
                ic = daily_rank_ic(g, sc, f"fwd_excess_{horizons[0]}d")
                if len(ic):
                    rob.append({"signal": sc, "split": key, "value": str(val), "mean_ic": float(ic.mean()),
                                "pct_positive": float((ic > 0).mean()), "n_dates": int(len(ic))})
    robust = pd.DataFrame(rob)
    robust.to_csv(out_dir / "robustness.csv", index=False)

    # 4. event study on accumulation scores
    days = trading_days(cfg)
    panel = read_dataset(cfg, "silver", "daily_panel", columns=["date", "isin", "adj_close_tr", "dq_exclude"])
    uni = read_dataset(cfg, "gold", "universe")
    er = excess_return_matrix(panel, uni, days)
    pool = df[["date", "isin", "med_traded_value", "vol_21d"]]
    ev_summary = {}
    car_tables = {}
    for model in ("lstm_ae", "iforest", "delivery_z_rule"):
        col = f"{model}__accumulation_score"
        if col not in df:
            continue
        events = select_events(df, col, cfg.anomaly.event_top_pct, min_gap=20, days=days)
        lo_w, hi_w = cfg.research.event_window
        res = event_study(events, er, pool, (lo_w, hi_w))
        if res["car"].empty:
            continue
        car_tables[model] = res["car"]
        res["car"].to_csv(out_dir / f"event_car_{model}.csv", index=False)
        ev_summary[model] = {"n_events": int(res["n_events"]["n"].iloc[0]),
                             "post": res["post"].to_dict(orient="records")}

    _plots(out_dir, ic_series, quint, car_tables, horizons[0])
    summary = {
        "period": period,
        "ic": ic_table.to_dict(orient="records"),
        "quintiles": quint.to_dict(orient="records"),
        "events": ev_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    best = ic_table[ic_table["horizon"] == horizons[0]].set_index("signal")["mean_ic"].round(4).to_dict()
    return {"period": period, "mean_ic_5d": best, "events": {k: v["n_events"] for k, v in ev_summary.items()}}


def _plots(out_dir, ic_series, quint, car_tables, h0) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (sc, h), ic in ic_series.items():
        if h == h0 and sc.endswith("__score"):
            ax.plot(ic.index, ic.cumsum(), label=sc.replace("__score", ""))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Cumulative daily rank IC vs {h0}d forward excess return (raw anomaly percentile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ic_cumulative.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    qcols = [c for c in quint.columns if c.startswith("Q")]
    x = np.arange(len(qcols))
    w = 0.8 / max(1, len(quint))
    for i, r in enumerate(quint.itertuples(index=False)):
        ax.bar(x + i * w, [getattr(r, c) * 1e4 for c in qcols], width=w, label=r.signal)
    ax.set_xticks(x + 0.4 - w / 2, qcols)
    ax.set_ylabel("mean fwd excess return (bps)")
    ax.set_title(f"{h0}-day forward excess return by score quintile")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "quintiles.png", dpi=120)
    plt.close(fig)

    if car_tables:
        fig, axes = plt.subplots(1, len(car_tables), figsize=(5 * len(car_tables), 4.2), sharey=True, squeeze=False)
        for ax, (model, car) in zip(axes[0], car_tables.items(), strict=False):
            for name, color in (("event", "C3"), ("control", "C0"), ("diff", "k")):
                ax.plot(car["rel_day"], car[f"{name}_car"] * 100, color=color, label=name)
                ax.fill_between(car["rel_day"], car[f"{name}_lo"] * 100, car[f"{name}_hi"] * 100, color=color, alpha=0.15)
            ax.axvline(0, color="grey", lw=0.8)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(f"{model}: top accumulation events")
            ax.set_xlabel("trading days from event")
        axes[0][0].set_ylabel("cumulative excess return (%)")
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "event_study.png", dpi=120)
        plt.close(fig)
