# NSE Smart-Money Anomaly Detector — Build Spec & Implementation Plan

> **One line:** apply fraud-style anomaly detection to Indian market "transaction" data (delivery volumes, bulk/block deals, price-volume behaviour) to flag unusual institutional accumulation, then test honestly whether those anomalies predict future returns.
>
> **How to use this file with Claude Code:** put it in the repo root (copy key rules into `CLAUDE.md`). Build **one phase at a time** ("Implement Phase 2 from NSE_SMART_MONEY_PLAN.md") and only move on when that phase's acceptance criteria pass. Review every diff — you must be able to defend every modelling and data decision in an interview.

---

## 0. Ground rules for the coding agent

1. **No lookahead, ever.** Every feature for date *t* may only use data published on or before *t*. This is enforced by automated tests (§9), not by good intentions.
2. **Point-in-time universes.** The stock universe on each date is built from data available on that date (no "today's Nifty 500" applied to the past).
3. **Raw data is immutable.** Downloaded files are stored untouched in `data/raw/`; all cleaning produces new versioned datasets in `data/processed/`.
4. **Every dataset is validated.** Schemas and sanity rules (pandera) run on every pipeline stage; violations fail the run loudly.
5. **Reproducible runs.** All randomness is seeded; every run writes a manifest (config, data snapshot hash, git commit, metrics) to `runs/<run_id>/`.
6. **Train/validation/test discipline.** Hyperparameters and thresholds are tuned only on validation folds. The final test period is touched **once**, at the end, and results are reported as-is.
7. **Report honestly.** Weak or negative results are valid findings. Never tune on the test period to improve the headline number.
8. **Config over code.** Dates, universe size, costs, horizons, and model settings live in `configs/*.yaml`.
9. **Keep it laptop-sized.** Daily data for ~500–800 stocks over ~10 years fits in DuckDB/Parquet on a laptop. GPU is optional (the LSTM autoencoder is small; Colab works if needed).

---

## 1. Research question & scope

**Research question.** *Do anomalies in India-specific market activity — unusual delivery volumes, bulk/block deal clustering, and abnormal price-volume behaviour — predict a stock's excess returns over the next 1–4 weeks, after realistic trading costs?*

**Hypothesis (to be tested, not assumed).** Genuine accumulation by large players leaves footprints: delivery % rises (shares are actually taken home, not day-traded), volume surges while price stays contained, and bulk deals cluster on the buy side. These footprints may precede price moves.

### In scope
- Data pipeline from NSE public archives (bhavcopy, full bhavcopy with delivery data, bulk/block deals, index data, corporate actions).
- Cleaning: corporate-action adjustment, symbol/ISIN mapping, data-quality anomaly checks.
- Point-in-time liquid universe (survivorship-bias-free).
- Feature engineering (momentum, volatility, liquidity, delivery, bulk-deal signals).
- **Anomaly models:** LSTM autoencoder (PyTorch) + Isolation Forest baseline.
- **Signal research:** event study and rank-IC analysis of anomaly scores.
- **Ranking model:** LightGBM with and without anomaly features (ablation = the research answer).
- Walk-forward validation with purging/embargo (skfolio's splitters).
- Cost-aware weekly backtest, portfolio construction with skfolio, benchmark comparison.
- Report + README with results, limitations, and future work.

### Out of scope
- Live/paper trading, broker integration, intraday data, options, short selling, leverage.
- Fundamental/alternative data beyond what NSE archives provide.

---

## 2. Data sources

| Dataset | Source | Notes |
|---|---|---|
| Equity bhavcopy (daily OHLCV, all NSE stocks) | NSE archives via **`aynse`** (maintained fork of jugaad-data) | Unadjusted prices |
| **Full bhavcopy** (includes **delivery quantity / delivery %**) | NSE archives via `aynse` (`full_bhavcopy_*`) | Core India-specific signal |
| Bulk & block deals | NSE via `aynse` (`bulk_deals_*`) | Named large trades, buy/sell side |
| Index bhavcopy (Nifty 500 TRI/PR) | NSE via `aynse` (`bhavcopy_index_*`) | Benchmark |
| Corporate actions (splits, bonuses, dividends) | NSE corporate-actions data (download/export) | Needed for price adjustment |
| ISIN ↔ symbol history | Bhavcopy ISIN column + NSE symbol-change list | Handles renames/mergers |

**Phase 0 must verify** for each dataset: (a) it downloads reliably, (b) how far back it goes, (c) schema changes over time (NSE changed the bhavcopy format around mid-2024 — the loader must handle old and new schemas). The study period is set by the **shortest** of the critical datasets (delivery data + bulk deals). Target ≥ 8 years; if bulk-deal history is shorter, use it only in a sub-period analysis.

**Etiquette:** throttle requests (1–2/sec), cache everything locally, never re-download a file that already exists, respect the source's terms.

---

## 3. Architecture

```
NSE archives
   │  ingest (aynse, throttled, cached, retries)
   ▼
data/raw/   (immutable files, by dataset/date)
   │  parse + normalize schemas (old & new formats)
   ▼
DuckDB / Parquet "bronze"  ── pandera validation
   │  ISIN mapping · corporate-action adjustment · DQ anomaly checks
   ▼
"silver": clean daily panel (one row per ISIN per day)
   │  point-in-time universe · features (t ≤ date only)
   ▼
"gold": feature panel + forward-return labels
   │
   ├─► Anomaly models (LSTM-AE, Isolation Forest) ─► anomaly scores
   │
   ├─► Signal research (event study, rank IC, quintiles)
   │
   └─► LightGBM ranker (± anomaly features) ─► weekly scores
           │
           ▼
      Portfolio construction (skfolio) ─► cost-aware backtest ─► report
```

### Tech stack
| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| Ingestion | `aynse`, `httpx` (retries/backoff) |
| Storage | Parquet + **DuckDB** |
| Dataframes | pandas (or Polars where speed matters) |
| Validation | **pandera** |
| Deep learning | **PyTorch** (LSTM autoencoder) |
| Classical ML | scikit-learn (Isolation Forest), **LightGBM** |
| CV & portfolios | **skfolio** (WalkForward, CombinatorialPurgedCV, optimizers) |
| Stats | scipy, statsmodels |
| CLI | Typer |
| Config | YAML + pydantic-settings |
| Plots/report | matplotlib/Plotly, Jinja2 → Markdown/HTML report |
| Testing | pytest, Hypothesis |
| CI | GitHub Actions, ruff, mypy |

---

## 4. Repository structure

```
nse-smart-money/
├── configs/
│   ├── base.yaml                # dates, universe size, horizons, seeds
│   ├── costs.yaml               # STT, brokerage, exchange, stamp, GST, slippage
│   └── models.yaml              # LSTM-AE, IsolationForest, LightGBM params
├── src/smd/
│   ├── cli.py                   # `smd ingest|clean|universe|features|anomaly|research|train|backtest|report`
│   ├── config.py
│   ├── ingest/                  # per-dataset downloaders + schema parsers
│   ├── clean/
│   │   ├── isin_map.py
│   │   ├── corporate_actions.py # adjustment factors
│   │   └── dq_checks.py         # data-quality anomaly detection
│   ├── universe/point_in_time.py
│   ├── features/
│   │   ├── price.py  ├── delivery.py  ├── bulk_deals.py  └── registry.py
│   ├── labels/forward_returns.py
│   ├── anomaly/
│   │   ├── lstm_ae.py           # model, dataset, training loop
│   │   ├── iforest.py
│   │   └── scoring.py           # reconstruction error → scores, accumulation filter
│   ├── research/
│   │   ├── event_study.py
│   │   └── ic_analysis.py
│   ├── model/ranker.py          # LightGBM, ablations
│   ├── cv/splits.py             # skfolio walk-forward + purge/embargo
│   ├── portfolio/construct.py   # equal weight, skfolio optimizers, caps
│   ├── backtest/
│   │   ├── engine.py            # weekly rebalance simulator
│   │   └── costs.py
│   ├── metrics/performance.py
│   ├── validation/schemas.py    # pandera schemas
│   └── report/build.py
├── tests/
│   ├── unit/ ├── integration/ └── fixtures/   # tiny sample dataset (≈20 stocks × 2 years)
├── notebooks/                   # exploration only; no logic lives here
├── data/ (raw/, processed/)     # gitignored
├── runs/                        # gitignored run manifests & artifacts
├── docs/ (CHANGELOG.md, DECISIONS.md, DATA.md)
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## 5. Data model (DuckDB / Parquet)

```
raw_bhavcopy(date, symbol, series, isin, open, high, low, close, prev_close,
             volume, traded_value, trades, source_file)
raw_delivery(date, symbol, series, isin, traded_qty, deliverable_qty, delivery_pct, source_file)
raw_bulk_deals(date, symbol, client_name, side, qty, price, deal_type, source_file)
  -- deal_type ∈ {'bulk','block'}; side ∈ {'BUY','SELL'}
corporate_actions(isin, symbol, ex_date, action_type, ratio_num, ratio_den, cash_amount)
  -- action_type ∈ {'split','bonus','dividend','rights','merger'}
isin_map(isin, symbol, valid_from, valid_to)

daily_panel(date, isin, symbol, open, high, low, close, adj_factor,
            adj_open, adj_high, adj_low, adj_close, volume, adj_volume,
            traded_value, deliverable_qty, delivery_pct, dq_flags)
universe(date, isin, in_universe BOOL, liquidity_rank)
features(date, isin, <feature columns…>)
labels(date, isin, fwd_ret_5d, fwd_ret_20d, fwd_excess_5d, fwd_excess_20d)
anomaly_scores(date, isin, model, score, accumulation_score, fold_id)
predictions(date, isin, model_variant, score, fold_id)
backtest_positions(rebalance_date, isin, weight)
backtest_daily(date, equity, gross_ret, net_ret, costs, turnover)
```

---

## 6. Component specifications

### 6.1 Ingestion
- One downloader per dataset, idempotent: skip files already present (checksum recorded in `data/raw/_manifest.parquet`).
- Retries with exponential backoff and jitter; throttle to ≤ 2 req/s; skip market holidays (use NSE holiday calendar; a missing file on a trading day is logged as a gap, not silently ignored).
- Parsers normalize **old and new** bhavcopy schemas into one canonical schema; unit-test both with fixture files.
- Keep only `series == 'EQ'` (and `BE` if needed, flagged) for equities.

### 6.2 Cleaning
**ISIN mapping.** Use ISIN as the stable stock identifier; maintain `isin_map` for symbol changes so a rename doesn't break a price series.

**Corporate-action adjustment.**
- Build a cumulative backward adjustment factor per ISIN from splits and bonuses: split 1:5 (face value 10 → 2) → factor 1/5 for all prior prices; bonus 1:1 → factor 1/2.
- `adj_price = raw_price × adj_factor`, `adj_volume = raw_volume / adj_factor`.
- Dividends: compute total-return series separately (`adj_close_tr`) for return labels; keep price-only series for price features.

**Data-quality (DQ) anomaly checks** — the first use of anomaly detection:
- Rule checks: `high ≥ max(open, close)`, `low ≤ min(open, close)`, `volume ≥ 0`, `0 ≤ delivery_pct ≤ 100`, `deliverable_qty ≤ traded_qty`.
- Statistical checks: daily |adjusted return| > 30% **without** a corporate action on that date → flag; robust z-score of log volume vs trailing 60 days > 8 → flag.
- Cross-check: a large unexplained jump that matches a typical split/bonus ratio (2:1, 5:1, 10:1, 1:1 bonus) → flag `possible_missing_corporate_action`.
- Flagged rows are excluded from features and reported in `docs/DATA.md` with counts. Every flag rule has a unit test.

### 6.3 Point-in-time universe (survivorship-bias-free)
- On each rebalance date, rank all EQ stocks by **trailing 126-day median traded value**, computed only from past data; take the top **N = 500** (configurable).
- Require ≥ 200 trading days of history and ≤ 5% missing days in the lookback.
- Stocks that later delist or crash stay in the historical universe as long as they qualified at the time — that is the point.
- Liquidity-based universes avoid needing historical index-constituent files, and are defensible: "the 500 most liquid stocks as of each date."

### 6.4 Features (all computed with data ≤ t)
Each feature lives in `features/registry.py` with name, function, lookback, and description.

**Price & volatility**
- Returns over 5, 21, 63, 126 days (skip the most recent 5 days for the 126-day momentum variant).
- Realized volatility 21d and 63d; downside volatility 63d.
- Distance from 52-week high; close location within the day's range (avg over 5d).
- Market beta 126d vs equal-weight universe.

**Liquidity**
- Log median traded value 21d; volume z-score vs 60d; Amihud illiquidity 21d.

**Delivery (India-specific — core novelty)**
- `delivery_pct` and its z-score vs the stock's own trailing 60d history.
- Delivery quantity z-score vs 60d (absolute delivered shares, not just %).
- Delivery-volume divergence: delivery z-score − volume z-score.
- "Quiet accumulation" flag: delivery z > 2 while |5d return| < 1 × 21d volatility.
- Cross-sectional rank of delivery z-score on the day.

**Bulk/block deals**
- Net bulk-deal buy value over 5d and 20d, normalized by 21d median traded value.
- Count of distinct buying entities over 20d; buy/sell imbalance.
- Days since last bulk buy (capped).

**Normalization:** cross-sectional rank-normalize features each date (to [-1, 1]) to remove market-wide level shifts; winsorize at 1st/99th percentiles.

**Timing convention (document in DECISIONS.md):** NSE publishes bhavcopy, delivery, and bulk-deal data after market close. Features are computed at close of day *t*; trades execute at **open of t+1** (configurable to close of t+1). Labels start from that execution price.

### 6.5 Labels
- `fwd_ret_5d`, `fwd_ret_20d`: total return from open(t+1) to open(t+1+h), from the total-return-adjusted series.
- `fwd_excess_*`: minus the equal-weight universe return over the same window.
- Labels near the end of the data where the window is incomplete are NaN and excluded.

### 6.6 Anomaly models

**A. LSTM autoencoder (PyTorch) — primary**
- Input: per-stock sequences of the last **L = 30** trading days of a compact behaviour vector (≈ 8–12 features): delivery % z, delivery qty z, volume z, return, range, close location, bulk-deal net value (normalized), turnover rank. Each sequence is standardized per stock using **only past data**.
- Architecture: encoder LSTM (2 layers, hidden 64) → bottleneck (16) → decoder LSTM → reconstruct the sequence. Loss: MSE (optionally Huber).
- Training: pooled across all stocks; sample sequences only from the training window of each walk-forward fold; early stopping on a validation slice inside the training window; seeds fixed; save checkpoint per fold.
- Score: reconstruction error of the latest day's sequence, converted to a **cross-sectional percentile** per date (so scores are comparable across time).
- Trained on mostly "normal" behaviour, high error means unusual behaviour.

**B. Isolation Forest — baseline**
- Same behaviour vector (latest day + 5d/20d summaries), fitted per fold on training data. Score → cross-sectional percentile.

**Accumulation filter (direction matters).** An anomaly can be distribution (selling) as easily as accumulation. Define:
`accumulation_score = anomaly_percentile × 1[delivery z > 0 AND net bulk buys ≥ 0 AND 5d return > −1σ]`
Report results for the raw anomaly score **and** the accumulation-filtered score.

### 6.7 Signal research (before any trading model)
Answer the research question directly:
1. **Rank IC:** each date, Spearman correlation between score and `fwd_excess_5d` / `fwd_excess_20d`. Report mean IC, IC standard deviation, IC information ratio, % of positive-IC dates, with block-bootstrap confidence intervals.
2. **Quintile analysis:** average forward excess return by score quintile; top-minus-bottom spread over time.
3. **Event study:** define events as accumulation_score in the top 2% on a day; plot average cumulative excess return from day −20 to +40 with confidence bands; compare with matched non-event stocks (same date, similar liquidity and volatility).
4. **Sub-periods & robustness:** results by year, by liquidity tercile, and for LSTM-AE vs Isolation Forest vs a simple delivery-z rule (to check the deep model adds anything over a one-line rule).

### 6.8 Ranking model (LightGBM) & ablation
- Target: `fwd_excess_5d` (regression on cross-sectional ranks) or LambdaRank grouped by date.
- **Variants (the ablation):**
  - `base`: price/volatility/liquidity features only.
  - `base+delivery`: + delivery and bulk-deal features.
  - `base+delivery+anomaly`: + anomaly and accumulation scores.
- Compare out-of-sample rank IC and backtest results across variants. **This comparison is the project's answer.**
- Feature importance via SHAP on validation folds (report, don't over-interpret).

### 6.9 Cross-validation
- **Walk-forward, expanding window:** retrain every 12 months; train on all data up to the cutoff, predict the next 12 months.
- **Purge + embargo:** remove training samples whose label window overlaps the test window, plus an embargo of the label horizon (e.g., 5 or 20 trading days). Use skfolio's splitters (`WalkForward`, `CombinatorialPurgedCV` for robustness checks) — reuse your skfolio contribution where it fits.
- Split the timeline: **development period** (all tuning) and a final **held-out test period** (last ~2 years) touched once.

### 6.10 Portfolio construction
- Weekly rebalance (configurable). Hold the top **K = 20–30** stocks by model score.
- Weighting: (a) equal weight; (b) skfolio optimizer (e.g., hierarchical risk parity or minimum variance) on the selected names using trailing 126d returns.
- Constraints: max 10% per stock, max 30% per sector (sector from NSE industry classification if available; otherwise skip the sector cap and document it), minimum liquidity (position ≤ 5% of 21d median traded value at a notional capital of ₹10 lakh, configurable).
- Turnover control: only trade names whose rank moved out of a buffer zone (e.g., sell when rank > 1.5K) to cut costs.

### 6.11 Backtest engine
- Daily mark-to-market between rebalances using adjusted total-return prices.
- Executes at the configured price (open t+1); fills capped by the liquidity constraint.
- **Costs** (`configs/costs.yaml`, verify current rates from NSE/your broker before the final run, since they change): STT on delivery buy and sell, brokerage, exchange transaction charges, SEBI fees, stamp duty on buys, GST on brokerage + charges, plus a **slippage model** (e.g., base 5 bps + impact term proportional to participation in daily traded value).
- Accounting invariants checked every day: `equity = cash + Σ holdings`, weights sum ≤ 1, costs ≥ 0, no trades in stocks outside the point-in-time universe, no position in a stock with no price that day.
- Delisting handling: if a held stock stops trading, exit at the last available price minus a haircut (configurable) and log it.

### 6.12 Metrics & benchmarks
- Benchmarks: **Nifty 500 index** (from index bhavcopy) and **equal-weight universe**.
- Metrics: CAGR, annualized volatility, **Sharpe**, Sortino, max drawdown, Calmar, turnover, average holdings, hit rate, beta/alpha vs Nifty 500, cost drag (gross − net return).
- Multiple-testing awareness: report how many variants were tried; compute the **deflated Sharpe ratio** for the chosen strategy.
- Report every variant from the ablation, not just the best.

### 6.13 Report
`smd report --run <id>` generates `reports/<run_id>/REPORT.md` (and HTML) containing: data coverage and DQ flag counts, universe size over time, IC tables and plots, quintile spreads, event-study chart, ablation table, equity curves vs benchmarks, drawdown plot, cost breakdown, limitations. Results in the README link to the latest report.

---

## 7. CLI

```
smd ingest     --datasets bhavcopy,delivery,bulk,index,corp --from 2014-01-01
smd clean
smd universe
smd features
smd labels
smd anomaly    --model lstm_ae|iforest --fold all
smd research
smd train      --variant base|base+delivery|base+delivery+anomaly
smd backtest   --variant ... --weighting equal|hrp
smd report     --run <run_id>
smd all        # full pipeline end to end
```

---

## 8. Configuration (`configs/base.yaml` excerpt)

```yaml
seed: 42
study_start: 2015-01-01          # set from Phase 0 coverage findings
dev_end: 2023-12-31
test_start: 2024-01-01
universe:
  size: 500
  liquidity_lookback_days: 126
  min_history_days: 200
labels:
  horizons: [5, 20]
  execution: next_open
cv:
  retrain_every_months: 12
  embargo_days: 20
portfolio:
  rebalance: weekly
  top_k: 25
  max_weight: 0.10
  notional_inr: 1000000
anomaly:
  seq_len: 30
  event_top_pct: 0.02
```

---

## 9. Testing & CI

**Lookahead tests (most important).**
- *Future-perturbation test:* for a random sample of (stock, date), randomly change all raw data **after** date *t*; recompute features for *t*; assert they are **identical**. Run for every registered feature, the universe builder, and anomaly scoring.
- *Label alignment test:* labels at *t* use only prices from t+1 onward.
- *CV leakage test:* no training sample's label window overlaps any test sample's window plus embargo.

**Data tests.**
- Corporate-action adjustment: synthetic split/bonus series produce continuous adjusted prices; a known real split (pick one from your data) is handled correctly.
- DQ rules: each rule flags a constructed bad row and passes a good one.
- Parser tests for old and new bhavcopy formats.
- pandera schemas on every stage.

**Model & backtest tests.**
- LSTM-AE: overfits a tiny batch (loss → near zero) — sanity check; output shapes; deterministic with fixed seed.
- Backtest invariants (property-based with Hypothesis on random weights/prices): equity accounting identity holds, costs ≥ 0, zero-signal strategy ≈ benchmark minus costs.
- Metrics: Sharpe/drawdown functions match hand-computed values on toy series.

**CI.** GitHub Actions runs ruff, mypy, and pytest on the small fixture dataset (≈ 20 stocks × 2 years, committed to `tests/fixtures/`) so CI never downloads from NSE. Coverage target ≥ 85% on `clean/`, `features/`, `universe/`, `backtest/`, `cv/`.

---

## 10. Phased build plan

Each phase: implement → tests green → update `docs/CHANGELOG.md` → commit.

### Phase 0 — Data availability spike (1–2 days)
- Script that downloads a few sample days from each dataset across different years (2012, 2016, 2020, 2024, 2026).
- Record coverage start dates, schema versions, and gaps in `docs/DATA.md`.
- **Accept:** a written decision on the study period and which datasets are usable. If delivery or bulk-deal history is too short, adjust scope **before** building anything else.

### Phase 1 — Skeleton & ingestion
- Repo structure, config, CLI, logging, run manifests.
- Idempotent, throttled downloaders + canonical parsers for all datasets; bronze tables in DuckDB.
- **Accept:** full history ingested; re-running ingestion downloads nothing new; parser tests pass for both schemas.

### Phase 2 — Cleaning & data quality
- ISIN mapping, corporate-action adjustment, total-return series, DQ anomaly checks, pandera schemas → silver `daily_panel`.
- **Accept:** adjustment tests pass; DQ report lists flag counts; a manual spot-check of 5 known splits/bonuses looks correct.

### Phase 3 — Universe, features, labels
- Point-in-time liquidity universe; feature registry with all §6.4 features; labels; gold tables.
- **Accept:** future-perturbation lookahead tests pass for every feature and the universe; universe size stable around N over time.

### Phase 4 — Anomaly models
- LSTM autoencoder (PyTorch) and Isolation Forest, trained per walk-forward fold; anomaly + accumulation scores stored.
- **Accept:** overfit-tiny-batch test passes; scores exist for every out-of-sample date; training is deterministic with a fixed seed.

### Phase 5 — Signal research
- Rank IC, quintiles, event study, robustness splits, comparison vs simple delivery-z rule.
- **Accept:** research section of the report generated with plots and bootstrap confidence intervals. Write a short honest conclusion in `docs/DECISIONS.md` (does the signal exist?).

### Phase 6 — Ranking model & ablation
- LightGBM variants with purged walk-forward CV; out-of-sample predictions; SHAP summary.
- **Accept:** CV leakage test passes; ablation table (IC per variant) generated on development folds only.

### Phase 7 — Portfolio construction & backtest
- Equal-weight and skfolio weighting, constraints, turnover buffer, cost model, backtest engine, metrics, benchmarks.
- **Accept:** backtest invariant tests pass; zero-signal sanity check behaves as expected; equity curves for all variants vs Nifty 500 and equal-weight universe.

### Phase 8 — Final test & report
- Freeze all choices; run once on the held-out test period; generate the full report.
- **Accept:** report includes development and test results side by side, deflated Sharpe, cost drag, limitations.

### Phase 9 — Polish
- README: problem, pipeline diagram, key charts, results table, limitations, how to reproduce (`smd all`), future work (intraday data, gamma-blast/options extension, live paper trading).
- **Accept:** a stranger can clone, run on the fixture dataset in minutes, and understand the findings from the README alone.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Delivery or bulk-deal history shorter than hoped | Phase 0 spike decides scope first; sub-period analysis if needed |
| NSE archive changes or throttling | Local cache, idempotent downloads, retries, schema-versioned parsers |
| Missing corporate actions corrupt prices | DQ check for unexplained split-like jumps; exclude flagged rows; document counts |
| Lookahead leaks inflate results | Automated future-perturbation and CV-leakage tests in CI |
| Overfitting via many variants | Development/test split, fixed variant list, deflated Sharpe, report all variants |
| Signal disappears after costs | Turnover buffer, liquidity caps, report gross vs net honestly — a negative result is still a credible finding |
| Deep model adds nothing over a simple rule | Explicit comparison vs delivery-z rule and Isolation Forest; report it either way |
| Too similar to common "LSTM predicts prices" projects | The novelty is the India-specific delivery/bulk-deal data and fraud-style anomaly framing — keep the README focused on that |

---

## 12. Resume mapping (fill in only with real results)

> **NSE Smart-Money Anomaly Detector — Fraud-Style ML on Market Transaction Data** github.com/abhib05/[repo]
> ● Built an end-to-end **ML pipeline** on **[N] years** of NSE data (**[M]M+** daily records): automated **data extraction**, corporate-action adjustment, **survivorship-bias-free** point-in-time universes, and **data-quality anomaly checks**, with automated **lookahead-bias tests** in CI.
> ● Trained a **PyTorch LSTM autoencoder** on India-specific **delivery-volume** and **bulk-deal** signals to flag unusual institutional accumulation; in **purged walk-forward validation**, anomaly features **[raised / did not raise]** LightGBM rank IC from **[a]** to **[b]**, and the top-**[K]** portfolio returned **Sharpe [X]** net of STT and brokerage vs **[Y]** for the Nifty 500.

Interview talking points this plan gives you: lookahead and survivorship bias and how you tested for them, corporate-action adjustment, point-in-time data, anomaly detection on transaction streams (the same shape as fraud detection), purged cross-validation, ablation design, cost-aware evaluation, and why you reported an honest (possibly modest) result.
