# CLAUDE.md

Project: **NSE Smart-Money Anomaly Detector**. Full spec: `docs/NSE_SMART_MONEY_PLAN.md`.

## Non-negotiable rules
1. **No lookahead.** A feature/score for date *t* may only use data published on or before *t*.
   Every new feature must pass the future-perturbation test in `tests/unit/test_lookahead.py`.
2. **Point-in-time universe** — built only from data available on each date.
3. **`data/raw/` is immutable.** Never edit or overwrite raw files; cleaning writes new datasets to `data/processed/`.
4. **Validate every stage** with the pandera schemas in `src/smd/validation/schemas.py`.
5. **Reproducible runs** — seeds fixed; each run writes `runs/<run_id>/manifest.json`.
6. **Train/validation/test discipline** — tune on development folds only; the test period
   (`test_start` onward) is evaluated once via `smd final-test`.
7. **Report honestly** — weak or negative results are valid findings.
8. **Config over code** — dates, sizes, costs, horizons, model params live in `configs/*.yaml`.

## Workflow
- Python env: `.venv` (`python -m venv .venv && pip install -e .[dev]`).
- Checks: `ruff check src tests`, `mypy src`, `pytest`.
- Run on the fixture: `smd all --config configs/fixture.yaml`.
- Record decisions in `docs/DECISIONS.md`, changes in `docs/CHANGELOG.md`.
