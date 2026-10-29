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

## D-010 — Timing convention (Phase 3)
NSE publishes bhavcopy, delivery (MTO) and bulk/block deals after the close. Features are
computed at the close of *t*; trades execute at the **open of t+1** (configurable to the
close of t+1); labels start from that execution price (`fwd_ret_h = P(t+1+h)/P(t+1) - 1`
on the total-return series). `label_end_h` is stored for purging.

## D-011 — Feature definitions (Phase 3)
* "delivery z" means the z-score of delivery % against the stock's own previous 60 days;
  z-scores always exclude day *t* from the mean/std.
* Quiet accumulation: delivery z > 2 while |5d return| < 21d daily vol x sqrt(5).
* Quantities are standardised in logs (`log(x)`, not `log1p`) so a backward split
  adjustment is an exact additive shift (keeps look-ahead tests exact to 1e-9).
* The anomaly channel `bh_turnover_rank` ranks traded value across **all** stocks traded that
  day. A first version ranked only among stocks that are ever in the universe — a set that
  depends on the future. The future-perturbation test caught it.
* Features are stored raw; models rank-normalise per date (winsorise 1/99%, rank to [-1, 1],
  missing -> 0).

## D-012 — Anomaly-model design (Phase 4)
* Behaviour vector (8 channels): delivery-% z, delivery-qty z, volume z, return z, range z
  (vs own trailing 252d), close location, daily net bulk-buy value / 21d median traded value,
  turnover rank. Clipped to ±8; missing cells -> 0; a window needs its last day observed and
  >= 80% of days observed.
* Walk-forward folds (skfolio `WalkForward`, expanding, yearly): first fold trains on 3 years.
  Each fold samples <= 120k training sequences from universe members; early stopping on the
  last 10% of the training window (time-ordered). Isolation Forest uses the same sample.
* Score = reconstruction MSE over the last 5 days of the 30-day window (localises the anomaly
  at *t*), converted to a cross-sectional percentile per date.
* Accumulation filter: percentile x 1[delivery z > 0, net bulk buys 20d >= 0, 5d return >
  -1 x 5-day vol]. The one-line baseline `delivery_z_rule` is scored with the same pipeline.

## D-013 — Research methodology (Phase 5)
* Daily rank IC with a moving-block bootstrap (block = label horizon) for the mean-IC CI,
  because overlapping h-day labels make daily ICs autocorrelated.
* Event study: top 2% accumulation scores per day; repeated events of the same stock within 20
  trading days are dropped; matched controls = same-day universe stocks in the same liquidity
  and volatility terciles; confidence bands cluster by event date.
* Research uses the development period only; the test period is evaluated once (Phase 8).
