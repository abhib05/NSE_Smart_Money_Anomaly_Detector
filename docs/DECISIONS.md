# Decisions

Each decision has an id, the choice, and the reasoning. New decisions are appended;
superseded ones are marked, never deleted.

## D-001 — Study period and test split (Phase 0)
Study 2015-01-01 → latest; development ≤ 2024-12-31; held-out test 2025-01-01 →
latest. All critical datasets reach back to 2012 (docs/DATA.md §1), so no sub-period
restriction is needed. The test boundary sits on a calendar year so that yearly
walk-forward folds never straddle it.

## D-002 — Download archive files directly, not through `aynse` parsers (Phase 0)
`aynse` returns already-parsed records (and its post-2024 bhavcopy route returned the
wrong file during the spike). Raw files must be stored byte-for-byte, so ingestion
fetches the same archive URLs with a throttled, retrying `httpx` client and keeps
`aynse` out of the critical path.

## D-003 — Delivery data from MTO files (Phase 0)
`sec_bhavdata_full_*.csv` (the "full bhavcopy") only exists from ~2020. The
security-wise delivery position file `MTO_ddmmyyyy.DAT` carries the same
traded/deliverable quantity and delivery % back to 2012 and is used instead.

## D-004 — Deals via `csv=true` (Phase 0)
The deals JSON API silently truncates at 70 rows (verified: 70 vs 162 rows on
2024-09-16). Monthly CSV downloads return the complete set.
