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
