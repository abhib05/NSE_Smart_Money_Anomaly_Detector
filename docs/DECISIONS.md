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

## D-005 — Stable security id = ISIN chain (Phase 2)
NSE usually issues a new ISIN on a face-value split, so raw ISIN breaks price series at
exactly the events we must adjust for. `clean/isin_map.py` links two ISINs when the same
symbol hands over from one to the other within 5 trading days; symbol changes keep the
ISIN. The chain's earliest ISIN is the security id (stored in the `isin` column of all
downstream tables; the raw ISIN is kept as `raw_isin`). The id is a label only.

## D-006 — Missing corporate actions inferred from the exchange's `prev_close` (Phase 2)
NSE's bhavcopy `PREVCLOSE` on an ex-date is already adjusted. If
`prev_close_t / close_{t-1}` deviates from 1 by >15% and matches a standard split/bonus/
consolidation ratio (±3%) while the corporate-action list has no event, the event is
inferred and flagged `ca_inferred_from_prevclose`. Listed events whose implied ratio
disagrees are flagged `ca_prevclose_mismatch`. Unexplained >30% jumps are flagged and
excluded.

## D-007 — Volume spikes are flagged, not excluded (Phase 2)
The plan's statistical DQ check flags log-volume robust z > 8. Such rows are reported but
kept: an extreme volume day is exactly the behaviour the anomaly models study. Only rule
violations (OHLC/volume/delivery invariants) and unexplained price jumps are excluded.

## D-008 — Backward adjustment and look-ahead (Phase 2)
Backward-adjusted prices before an ex-date are rescaled when the event happens. All
features use ratios within a window (returns, z-scores of log quantities, price/rolling
max), which are invariant to a constant rescaling of every point in the window, so a
future corporate action cannot change a feature value at *t* (up to floating-point
rounding — the look-ahead tests compare with a 1e-9 relative tolerance). Traded value
(₹) is never adjusted.

## D-009 — Total-return series for labels (Phase 2)
Dividends are applied as `1 - D / prev_close` on the ex-date to a separate `tr_factor`;
labels and backtest marking use `adj_open_tr` / `adj_close_tr`, price features use the
price-only `adj_*` columns.
