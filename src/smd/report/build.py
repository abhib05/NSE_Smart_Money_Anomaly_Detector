"""`smd report --run <id>` -> `reports/<run_id>/REPORT.md` + `REPORT.html`.

Collects everything a run produced (data coverage, DQ flags, universe, research, ablation,
backtests for development and test periods) into one document. Missing sections (e.g.
before the final test) are reported as pending rather than silently omitted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import markdown  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

from smd.config import Config  # noqa: E402
from smd.final_test import previous_evaluations  # noqa: E402
from smd.utils import dataset_exists, get_logger, read_dataset  # noqa: E402

log = get_logger(__name__)


def md_table(df: pd.DataFrame | None, floatfmt: str = "{:.4f}", pct_cols: tuple[str, ...] = ()) -> str:
    if df is None or df.empty:
        return "_not available_"

    def fmt(c, v):
        if isinstance(v, float | np.floating):
            if np.isnan(v):
                return "–"
            return f"{100 * v:.2f}%" if c in pct_cols else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row, strict=False)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


TEMPLATE = """# NSE Smart-Money Anomaly Detector — Report

Run `{{ run_id }}` · git `{{ git }}` · config `{{ fingerprint }}` · data snapshot `{{ snapshot }}`

**Research question.** Do anomalies in India-specific market activity — unusual delivery
volumes, bulk/block-deal clustering and abnormal price-volume behaviour — predict a stock's
excess return over the next 1–4 weeks, after realistic trading costs?

Development period: {{ study_start }} → {{ dev_end }} (all tuning). Held-out test period:
{{ test_start }} → end of data, evaluated **{{ n_test_evals }}** time(s) with this configuration.

## 1. Data coverage

{{ coverage }}

### Data-quality flags

{{ dq }}

![universe size](universe_size.png)

## 2. Signal research (anomaly scores)

Rank IC = daily Spearman correlation between the score and the forward excess return;
95% CIs from a moving-block bootstrap (block = horizon).

### Development period

{{ ic_dev }}

Quintiles ({{ h0 }}-day forward excess return, mean per date):

{{ q_dev }}

![IC](research_dev_ic_cumulative.png)
![quintiles](research_dev_quintiles.png)
![event study](research_dev_event_study.png)

Event study — post-event excess return vs matched controls (cluster-by-date SE):

{{ ev_dev }}

### Test period

{{ ic_test }}

{{ q_test }}

![event study test](research_test_event_study.png)

## 3. Ranking model ablation (LightGBM, purged walk-forward)

Development folds:

{{ abl_dev }}

Combinatorial purged CV on the development period (mean IC across paths):

{{ cpcv }}

Test period:

{{ abl_test }}

Top features by mean |SHAP| (development folds, `base+delivery+anomaly`):

{{ shap }}

## 4. Backtests (weekly rebalance, net of Indian delivery costs)

### Development period

{{ bt_dev }}

![equity dev](backtest_dev_equity_curves.png)

### Test period

{{ bt_test }}

![equity test](backtest_test_equity_curves.png)

### Cost breakdown (₹, development period)

{{ costs_dev }}

`deflated_sharpe` = probability that the strategy's true Sharpe beats the best Sharpe expected
from {{ n_trials }} unskilled trials (Bailey & López de Prado). Benchmarks: the Nifty 500
*price* index (understates a total-return benchmark by roughly its dividend yield) and an
equal-weight total-return portfolio of the point-in-time universe without costs.

## 5. Limitations

- Corporate actions come from NSE's free-text announcements; splits/bonuses missing from that
  list are inferred from the exchange's adjusted previous close, and unexplained jumps are excluded.
- No point-in-time sector classification is available from NSE archives, so the 30% sector cap is
  not applied.
- Weekend special sessions (Muhurat trading, Budget-day Saturdays) are not ingested.
- Execution at the next open assumes fills at the official open price; slippage is modelled, but
  queue position and auction dynamics are not.
- Delisted holdings exit at the last price minus a {{ haircut }} haircut; labels for delisted
  stocks use the last available price.
- The Nifty 500 benchmark is a price index (no dividends).
- Costs use 2025-26 published rates; rates have changed over the study period.
- Daily ICs of overlapping multi-day labels are autocorrelated; the block bootstrap mitigates but
  does not remove this.
"""


def _universe_plot(cfg: Config, out: Path) -> None:
    if not dataset_exists(cfg, "gold", "universe"):
        return
    u = read_dataset(cfg, "gold", "universe", columns=["date", "in_universe"])
    s = u[u["in_universe"]].groupby("date").size()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(s.index, s.values)
    ax.axvline(pd.Timestamp(cfg.test_start), color="r", ls="--", lw=1, label="test start")
    ax.set_title("Point-in-time universe size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "universe_size.png", dpi=110)
    plt.close(fig)


def _coverage(cfg: Config) -> str:
    rows = []
    if dataset_exists(cfg, "silver", "daily_panel"):
        p = read_dataset(cfg, "silver", "daily_panel", columns=["date", "isin", "delivery_pct"])
        rows += [("daily panel rows (EQ series)", f"{len(p):,}"), ("securities", f"{p['isin'].nunique():,}"),
                 ("trading days", f"{p['date'].nunique():,}"),
                 ("date range", f"{p['date'].min():%Y-%m-%d} → {p['date'].max():%Y-%m-%d}"),
                 ("rows with delivery data", f"{p['delivery_pct'].notna().mean():.1%}")]
    if dataset_exists(cfg, "silver", "deals"):
        d = read_dataset(cfg, "silver", "deals", columns=["deal_type"])
        rows += [(f"{k} deals", f"{v:,}") for k, v in d["deal_type"].value_counts().items()]
    if dataset_exists(cfg, "silver", "corporate_actions"):
        c = read_dataset(cfg, "silver", "corporate_actions", columns=["action_type"])
        rows += [(f"corporate actions: {k}", f"{v:,}") for k, v in c["action_type"].value_counts().items()]
    if dataset_exists(cfg, "gold", "universe"):
        u = read_dataset(cfg, "gold", "universe", columns=["isin", "in_universe"])
        rows.append(("securities ever in the universe", f"{u.loc[u['in_universe'], 'isin'].nunique():,}"))
    return md_table(pd.DataFrame(rows, columns=["item", "value"]))


def _ic(path: Path) -> pd.DataFrame | None:
    t = _csv(path)
    if t is None:
        return None
    t = t[["signal", "horizon", "n_dates", "mean_ic", "ci_low", "ci_high", "ic_ir", "pct_positive"]]
    return t.sort_values(["horizon", "mean_ic"], ascending=[True, False])


def _events(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8")).get("events", {})
    rows = []
    for model, v in s.items():
        for r in v["post"]:
            rows.append({"model": model, "n_events": v["n_events"], "horizon_days": int(r["horizon"]),
                         "excess_vs_controls": r["mean"], "se": r["se"]})
    return pd.DataFrame(rows) if rows else None


def _abl(path: Path) -> pd.DataFrame | None:
    t = _csv(path)
    if t is None:
        return None
    cols = [c for c in ["variant", "n_features", "n_dates_5d", "mean_ic_5d", "ci_low_5d", "ci_high_5d", "ic_ir_5d",
                        "mean_ic_20d", "ci_low_20d", "ci_high_20d"] if c in t]
    return t[cols]


def _bt(path: Path) -> pd.DataFrame | None:
    t = _csv(path)
    if t is None:
        return None
    cols = [c for c in ["strategy", "cagr", "gross_cagr", "cost_drag", "ann_vol", "sharpe", "sortino", "max_drawdown",
                        "calmar", "annual_turnover", "beta", "alpha", "hit_rate_weekly", "deflated_sharpe"] if c in t]
    return t[cols]


def build_report(cfg: Config, run) -> Path:
    out = cfg.paths.reports_dir / run.run_id
    out.mkdir(parents=True, exist_ok=True)
    m = run.load_manifest()
    rd = run.dir
    for sub in ("research_dev", "research_test", "backtest_dev", "backtest_test"):
        for png in (rd / sub).glob("*.png") if (rd / sub).exists() else []:
            shutil.copy(png, out / f"{sub}_{png.name}")
    _universe_plot(cfg, out)
    dq = _csv(cfg.paths.reports_dir / "dq_summary.csv")
    if dq is None and dataset_exists(cfg, "silver", "dq_summary"):
        dq = read_dataset(cfg, "silver", "dq_summary")
    shap = _csv(rd / "model" / "shap_dev.csv")
    if shap is not None:
        shap = shap[shap["variant"] == "base+delivery+anomaly"].head(15)
    btd = _bt(rd / "backtest_dev" / "metrics.csv")
    n_trials = int(btd["deflated_sharpe"].notna().sum()) if btd is not None and "deflated_sharpe" in btd else 0
    pct = ("cagr", "gross_cagr", "cost_drag", "ann_vol", "max_drawdown", "alpha", "hit_rate_weekly")
    ctx = dict(
        run_id=run.run_id, git=m.get("git_commit", "?")[:12], fingerprint=m.get("config_fingerprint", "?"),
        snapshot=m.get("data_snapshot", "?"), study_start=cfg.study_start, dev_end=cfg.dev_end,
        test_start=cfg.test_start, n_test_evals=len([e for e in previous_evaluations(cfg)
                                                    if e["config_fingerprint"] == cfg.fingerprint()]),
        coverage=_coverage(cfg), dq=md_table(dq, "{:.4f}"), h0=cfg.labels.horizons[0],
        ic_dev=md_table(_ic(rd / "research_dev" / "ic_table.csv")),
        q_dev=md_table(_csv(rd / "research_dev" / "quintiles.csv"), "{:.5f}"),
        ev_dev=md_table(_events(rd / "research_dev" / "summary.json"), "{:.4f}"),
        ic_test=md_table(_ic(rd / "research_test" / "ic_table.csv")) if (rd / "research_test").exists()
        else "_pending — run `smd final-test`_",
        q_test=md_table(_csv(rd / "research_test" / "quintiles.csv"), "{:.5f}") if (rd / "research_test").exists() else "",
        abl_dev=md_table(_abl(rd / "model" / "ablation_dev.csv")),
        cpcv=md_table(_csv(rd / "model" / "cpcv_dev.csv")),
        abl_test=md_table(_abl(rd / "model" / "ablation_test.csv")) if (rd / "model" / "ablation_test.csv").exists()
        else "_pending — run `smd final-test`_",
        shap=md_table(shap, "{:.5f}"),
        bt_dev=md_table(btd, "{:.3f}", pct),
        bt_test=md_table(_bt(rd / "backtest_test" / "metrics.csv"), "{:.3f}", pct)
        if (rd / "backtest_test").exists() else "_pending — run `smd final-test`_",
        costs_dev=md_table(_cost_table(rd / "backtest_dev" / "cost_breakdown.csv"), "{:,.0f}"),
        n_trials=n_trials, haircut=f"{cfg.backtest.delisting_haircut:.0%}",
    )
    text = Template(TEMPLATE).render(**ctx)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables"])
    (out / "REPORT.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>NSE Smart-Money Report</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "table{border-collapse:collapse;font-size:.85rem;margin:1rem 0}td,th{border:1px solid #ccc;padding:.25rem .5rem}"
        "img{max-width:100%}</style></head><body>" + html + "</body></html>", encoding="utf-8")
    return out / "REPORT.md"


def _cost_table(p: Path) -> pd.DataFrame | None:
    t = _csv(p)
    if t is None:
        return None
    t = t.rename(columns={t.columns[0]: "strategy"})
    t["total"] = t.drop(columns="strategy").sum(axis=1)
    return t
