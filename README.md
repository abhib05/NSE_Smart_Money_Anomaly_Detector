# NSE Smart-Money Anomaly Detector

**Fraud-style anomaly detection on Indian market "transaction" data.** The project looks for
unusual institutional accumulation in NSE stocks: delivery volumes (shares actually taken
into demat, not day-traded), bulk/block deals, and abnormal price-volume behaviour. It then
tests honestly whether those anomalies predict excess returns over the next 1–4 weeks,
after realistic Indian trading costs.

> **Research question.** Do anomalies in India-specific market activity predict a stock's
> excess return over the next 1–4 weeks, after costs?

<!-- RESULTS:START -->
**Answer (held-out test 2025-01 → 2026-10, evaluated once).** Yes, delivery data predicts
1–4-week excess returns, but only weakly, and the edge does not survive Indian trading
costs at a weekly horizon. The fraud-style LSTM autoencoder adds nothing beyond a one-line
delivery-z rule. Full report: [`reports/latest/REPORT.md`](reports/latest/REPORT.md).
<!-- RESULTS:END -->

## Why this is interesting

Most "LSTM predicts stock prices" projects fit price series. This one treats the market like
a transaction stream in fraud detection. It learns what *normal* trading behaviour looks
like for each stock and flags departures from it. The inputs are India-specific
disclosures that most global datasets don't have:

| Signal | Source | Intuition |
|---|---|---|
| Delivery % and delivered quantity | NSE security-wise delivery position (`MTO_*.DAT`) | Accumulators take delivery; speculators square off intraday |
| Bulk & block deals (named client, side) | NSE deals archive | Large disclosed buyers clustering on the buy side |
| Volume surges with contained price | bhavcopy | "Quiet accumulation" |

## Pipeline

```
NSE archives ──ingest (throttled, cached, checksummed)──► data/raw/  (immutable)
   │ parse old + UDiFF bhavcopy, MTO delivery, deals, corporate actions, index
   ▼
bronze (pandera-validated parquet + DuckDB views)
   │ ISIN chains · corporate-action + total-return adjustment · DQ anomaly checks
   ▼
silver: daily_panel (one row per security per day)
   │ point-in-time liquidity universe · 25 features (t ≤ date only) · labels (open t+1 →)
   ▼
gold ─┬─► anomaly models (PyTorch LSTM autoencoder, Isolation Forest, delivery-z rule)
      ├─► signal research (rank IC + block bootstrap, quintiles, event study vs matched controls)
      └─► LightGBM ranker ablation: base → +delivery/bulk → +anomaly  (purged walk-forward)
              ▼
         weekly portfolio (equal / skfolio HRP, caps, turnover buffer)
              ▼
         cost-aware backtest (STT, stamp, GST, DP, slippage) ─► report (dev + held-out test)
```

## Methodological guard-rails (all enforced by tests)

- **No look-ahead.** A *future-perturbation* test randomly rewrites all raw data after date
  *t*, including prices, volumes, delivery, deals and a new corporate action. It then rebuilds
  the pipeline and asserts that every feature, behaviour channel and the universe at dates ≤ *t*
  are unchanged. It caught a real leak during development (D-011 in
  [DECISIONS.md](docs/DECISIONS.md)).
- **Survivorship-bias-free universe.** The universe holds the 500 most liquid stocks *as of
  each date*, from trailing traded value. Delisted stocks stay in the past.
- **Corporate actions.** ISINs are chained across face-value splits, which change the ISIN.
  Prices are back-adjusted for splits and bonuses, and dividends go into a separate
  total-return series. Splits missing from NSE's list are inferred from the exchange's adjusted
  previous close, and unexplained jumps are flagged and excluded.
- **Purged walk-forward CV.** Folds come from skfolio `WalkForward`, with purging of
  overlapping labels and a 20-day embargo. A CV-leakage test checks the ranker's actual
  training rows, and CombinatorialPurgedCV serves as a robustness check.
- **One look at the test set.** `smd final-test` evaluates the frozen pipeline on 2025-01 →
  latest. Every evaluation is logged in `runs/TEST_LOG.jsonl`, and a repeat is refused unless
  forced. The report states how many times the test period was evaluated.
- **Multiple testing.** All 6 strategy variants are reported, along with the deflated Sharpe
  ratio.

<a id="results"></a>
## Results

<!-- RESULTS-DETAIL:START -->
Data: 2014-01 → 2026-10, 4.9M stock-days, 3,161 NSE equities, a point-in-time
universe of the 500 most liquid stocks, 144k bulk/block deals and 780 split/bonus adjustments.
Development 2015–2024; held-out test 2025-01-03 → 2026-10-29.

**1. Signal research: rank IC vs 5-day forward excess return (95% block-bootstrap CI)**

| Signal | Development | Test |
|---|---|---|
| Delivery-% z rule (one line) | **0.031** (0.027–0.036) | **0.029** (0.023–0.035) |
| LSTM autoencoder, raw anomaly | −0.016 (−0.022 – −0.009) | −0.004 (−0.014–0.006) |
| LSTM autoencoder + accumulation filter | 0.017 (0.013–0.021) | 0.013 (0.006–0.019) |
| Isolation Forest + accumulation filter | 0.016 (0.011–0.020) | 0.010 (0.003–0.017) |

Raw anomalies on their own predict *under*-performance; the direction filter turns them
positive, but they still trail the simple rule.

**2. Ranker ablation (LightGBM, purged walk-forward): 5-day out-of-sample rank IC**

| Variant | Dev 2020–24 | Test 2025–26 |
|---|---:|---:|
| base (price, volatility, liquidity) | 0.043 | 0.058 |
| + delivery & bulk/block deals | 0.052 | **0.069** |
| + LSTM-AE / IF anomaly scores | **0.054** | 0.066 |

**3. Backtest: top 25, weekly rebalance, equal weight, net of STT/stamp/GST/DP/slippage**

| | Dev net CAGR | Dev Sharpe | Test net CAGR | Test gross CAGR | Cost drag | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| base | 10.9% | 0.31 | −7.3% | 6.2% | ~13%/yr | 30–36× |
| + delivery | 18.4% | 0.64 | −0.3% | 16.5% | ~17%/yr | 35–41× |
| + delivery + anomaly | 19.3% | 0.69 | −1.8% | 15.0% | ~17%/yr | 36–42× |
| Nifty 500 (price index) | 18.3% | 0.65 | −2.2% | — | — | — |
| EW universe (TR, no costs) | 30.7% | 1.08 | 0.1% | — | — | — |

Sharpe ratios use a 6.5% risk-free rate. All 6 variant × weighting trials are reported in the
report, with deflated Sharpe ratios.

**Takeaways** ([DECISIONS.md D-020](docs/DECISIONS.md)):
- Delivery features add real out-of-sample information: about +10 points of gross CAGR.
- A 5-day horizon turns the book over about 40× a year, and costs consume about 17% a year.
- The deep anomaly detector is not worth its complexity here.
- Next steps: lower-turnover variants, such as a 20-day target or monthly rebalancing. These
  were deliberately *not* tuned, to keep the held-out test clean.

![Test-period equity curves](reports/latest/backtest_test_equity_curves.png)
![Event study](reports/latest/research_dev_event_study.png)
<!-- RESULTS-DETAIL:END -->

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -c requirements-lock.txt -e ".[dev]"   # tested versions (pandas 2.x)
```

**On the committed synthetic fixture (no downloads, ~1–2 minutes):**

```bash
smd all --config configs/fixture.yaml
```

This builds everything from `tests/fixtures/bronze` (24 synthetic stocks × 3 years, with
planted splits, an ISIN change, a delisting, bad prints and accumulation events). The report
lands in `reports/fixture/<run_id>/REPORT.md`.

**On real NSE data:** the full ingest downloads about 10k files at 2 requests/s, which takes
roughly 1.5 hours and about 1 GB:

```bash
export SMD_DATA_DIR=/path/with/space     # optional; default ./data
smd all
```

Or run it stage by stage:

```bash
smd ingest --from 2014-01-01      # idempotent: re-running downloads nothing new
smd clean                         # silver panel + DQ report (reports/dq_summary.md)
smd universe && smd features && smd labels
smd anomaly                       # walk-forward LSTM-AE + Isolation Forest
smd research                      # development period only
smd train                         # LightGBM ablation (dev folds)
smd backtest                      # all variants x weightings, dev period
smd final-test                    # held-out test period - once
smd report                        # reports/<run_id>/REPORT.md + .html
smd spotcheck INFY 2018-09-04     # raw vs adjusted prices around a corporate action
smd db                            # DuckDB views over every dataset
```

Checks: `ruff check src tests`, `mypy`, `pytest` (CI runs them on the fixture).

## Repository map

| Path | What |
|---|---|
| `configs/` | `base.yaml` (dates, universe, horizons, CV), `costs.yaml`, `models.yaml`, `fixture.yaml` |
| `src/smd/ingest/` | throttled client, idempotent downloaders, old/UDiFF parsers, bronze |
| `src/smd/clean/` | ISIN chains, corporate actions, DQ checks, silver panel |
| `src/smd/universe/`, `features/`, `labels/` | point-in-time universe, feature registry, forward returns |
| `src/smd/anomaly/` | LSTM autoencoder, Isolation Forest, scoring + accumulation filter |
| `src/smd/research/` | rank IC, quintiles, event study |
| `src/smd/model/`, `cv/` | LightGBM ranker + ablation, walk-forward / CPCV splits |
| `src/smd/portfolio/`, `backtest/`, `metrics/` | construction, cost model + engine, metrics + deflated Sharpe |
| `docs/` | [DATA.md](docs/DATA.md), [DECISIONS.md](docs/DECISIONS.md), [CHANGELOG.md](docs/CHANGELOG.md), the [build plan](docs/NSE_SMART_MONEY_PLAN.md) |

## Limitations

See the report's limitations section. The main ones:
- Corporate actions come from free-text NSE announcements.
- There is no point-in-time sector data, so no sector cap is applied.
- The Nifty 500 benchmark is a price index, not total return.
- Costs use current published rates.
- Weekend special sessions are not ingested.

## Future work

- Intraday / order-book data, to separate block-window prints from continuous trading.
- An options extension: unusual F&O open interest alongside delivery spikes ("gamma blasts").
- Live paper trading of the frozen strategy, to accumulate a true out-of-sample record.
- Client-level deal graphs: repeat buyers, and links between entities.
