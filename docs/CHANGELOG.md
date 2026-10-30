# Changelog

## Phase 0 — Data availability spike
- `scripts/phase0_spike.py` probes every dataset across 2012–2026.
- Findings and study-period decision recorded in `docs/DATA.md`; decisions D-001…D-004.

## Phase 1 — Skeleton & ingestion
- Package `smd` (src layout), `pyproject.toml`, typed YAML config (`configs/base|costs|models.yaml`, deep-merged overrides, `SMD_*_DIR` env overrides).
- Logging, seeding, run manifests (`runs/<run_id>/manifest.json`: config, fingerprint, git commit, data snapshot hash, per-stage metrics).
- Throttled (≤2 req/s), retrying (exponential backoff + jitter) HTTP client with NSE cookie priming.
- Idempotent downloaders for bhavcopy (old CSV + UDiFF), MTO delivery, index snapshots, bulk/block deals (monthly `csv=true`), corporate actions (quarterly), symbol changes. Checksums in `data/raw/_manifest.parquet`; raw files never overwritten; incomplete windows stored as `_asof<date>` files.
- Canonical parsers for both bhavcopy schemas + bronze builder with pandera validation and a derived trading calendar (gap detection).
- DuckDB views over all processed parquet (`smd db`). CLI: `smd ingest|bronze|db|new-run`.
- Tests: parsers on truncated real NSE files (old and UDiFF), mocked-NSE idempotency/404/retry/immutability tests.

## Phase 2 — Cleaning & data quality
- ISIN chains (`clean/isin_map.py`): stable security ids across face-value-split ISIN changes and symbol changes.
- Corporate actions (`clean/corporate_actions.py`): parser for NSE free-text subjects (split, consolidation, bonus, dividend incl. %-of-face-value, rights, merger); backward adjustment factors; total-return factor; missing splits/bonuses inferred from the exchange's adjusted prev_close.
- DQ checks (`clean/dq_checks.py`): OHLC/volume/delivery rules, unexplained >30% moves, split-like jumps, robust volume z; flag counts written to `reports/dq_summary.md`.
- Silver tables: `daily_panel`, `isin_map`, `corporate_actions`, `deals` (mapped to security ids), `benchmark`, `dq_summary`; pandera schemas.
- Deterministic synthetic fixture (`smd/fixture.py`, `tests/fixtures/bronze`, `configs/fixture.yaml`) with planted splits, ISIN change, symbol change, delisting, bad prints and accumulation events.
- CLI: `smd clean`, `smd spotcheck SYMBOL DATE`. Decisions D-005…D-009.

## Phase 3 — Universe, features, labels
- Point-in-time liquidity universe (top N by trailing 126d median traded value, ≥200 days history, ≤5% missing).
- Feature registry (24 features + cross-sectional delivery rank) in price / liquidity / delivery / bulk groups, plus 8 anomaly behaviour channels.
- Forward-return labels (open t+1 → open t+1+h on total-return prices, excess vs EW universe, label end dates).
- Future-perturbation look-ahead tests for every feature, behaviour channel and the universe; label alignment tests. The tests caught a real leak in `bh_turnover_rank` (fixed, D-011).
- CLI: `smd universe|features|labels`.

## Phase 4 — Anomaly models
- Walk-forward folds from skfolio `WalkForward` with purge/embargo helpers and a CombinatorialPurgedCV wrapper (`cv/splits.py`).
- PyTorch LSTM autoencoder (2x64 LSTM, 16-d bottleneck), seeded and early-stopped; per-fold checkpoints.
- Isolation Forest baseline on behaviour summaries; one-line delivery-z rule baseline.
- Cross-sectional percentile scores + accumulation filter → `gold/anomaly_scores`.
- Tests: overfit-tiny-batch, shapes, determinism, anomaly separation, window look-ahead, OOS coverage, planted-event sanity, CV leakage.

## Phase 5 — Signal research
- Rank IC with block-bootstrap CIs, IC IR, % positive; quintile spreads; event study with matched controls and date-clustered bands; robustness by year and liquidity tercile; LSTM-AE vs Isolation Forest vs delivery-z rule.
- `smd research` writes tables and charts to `runs/<id>/research_dev/`.

## Phase 6 — Ranking model & ablation
- LightGBM ranker, three fixed variants, purged walk-forward CV, TreeSHAP importances, CPCV robustness; ablation table on development folds only.
- Tests: CV leakage on the ranker's actual training rows, determinism, nested variants.

## Phase 7 — Portfolio construction & backtest
- Selection with turnover buffer, equal-weight and skfolio HRP weighting, max-weight and liquidity caps (water-filling).
- Indian delivery cost model (STT, stamp duty, exchange, SEBI, GST, DP charges) + square-root slippage.
- Daily backtest engine with next-open execution, liquidity-capped fills, delisting exits and per-day accounting invariants.
- Metrics incl. Sortino, Calmar, beta/alpha vs Nifty 500, weekly hit rate, cost drag, deflated Sharpe ratio; benchmarks Nifty 500 and EW universe; random-signal sanity strategy.
- Tests: Hypothesis property tests of invariants on random markets, zero-signal ≈ EW benchmark, delisting exit, hand-computed metrics.

## Phase 8 — Full real-data run, held-out test & report
- `smd final-test`: evaluates the frozen pipeline once on the test period; every evaluation is logged to `runs/TEST_LOG.jsonl` and repeats are refused unless `--force`.
- `smd report`: `reports/<run_id>/REPORT.md` + `REPORT.html` with data coverage, DQ flags, universe size, IC/quintile/event-study results, ablation (dev + test), CPCV, SHAP, backtests (dev + test), deflated Sharpe, cost breakdown and limitations.
- `smd all` runs every stage in its own process (bounded memory on a 16 GB laptop) and ends with the final test and the report; end-to-end CLI test on the fixture.
- Real-data fixes found by the first full ingest: niftyindices `dd/mm/yyyy` dates and zero closes, garbled MTO header labels, a 2-digit-year bhavcopy, NSE's many split phrasings (SBIN 2014, JSWSTEEL 2017), ETFs in the EQ series (now excluded).
- D-018: measured that NSE's `PREVCLOSE` is never adjusted on ex-dates — prev_close inference disabled, `ca_price_mismatch` flag added (supersedes D-006). D-017 LSTM compute budget, D-019 equity-only panel and demerger limitation.
- Corporate-action spot-check on six large caps and ingest statistics recorded in `docs/DATA.md`.
- Vectorised DQ hot paths and in-place cleaning for the 4.9M-row panel.

## Phase 9 — Polish
- README: problem, pipeline diagram, methodological guard-rails, results tables (dev + held-out test), charts, reproduction on the fixture and on real data, repository map, limitations, future work.
- GitHub Actions CI: ruff, mypy, pytest on the committed fixture (never downloads from NSE), coverage gates (≥85% overall and on clean/features/universe/backtest/cv).
