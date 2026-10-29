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
