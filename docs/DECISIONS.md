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

## D-014 — Ranker design and fixed variant list (Phase 6)
* Three variants fixed before looking at results: `base`, `base+delivery`,
  `base+delivery+anomaly`. Their comparison is the project's answer.
* Rows: weekly rebalance dates (last trading day of the week) where all variants have inputs.
  The anomaly scores are themselves out-of-sample, so the ranker's first fold starts 2 years
  after the first anomaly fold.
* Target: cross-sectional rank of `fwd_excess_5d`; LightGBM regression, deterministic mode.
* Purge (label end >= test start) + 20-day embargo on every fold; CombinatorialPurgedCV
  (6 folds, 2 test folds) as a robustness check on the development period.
* SHAP from LightGBM's built-in TreeSHAP on development folds only.

## D-015 — Backtest conventions (Phase 7)
* Weekly rebalance; top 25 names; held names kept while rank <= 1.5 x 25; kept names resized
  only if weight drifts > 25% from target; orders < ₹2,000 skipped (fixed DP charges) except
  full exits. Hypothesis found the case that motivated the last rule: a tiny resize-sell whose
  DP charge exceeded its proceeds and drove cash negative.
* Each order capped at 5% of that day's traded value; position cap 5% of 21d median traded
  value at the notional capital (₹10 lakh) and 10% of equity.
* Delisted / stale (> 5 days without a price) holdings exit at the last price minus 10%.
* Sharpe uses a 6.5% risk-free rate (approx. 91-day T-bill). Deflated Sharpe counts the 6
  variant x weighting trials; the random-signal strategy is a sanity check, not a trial.

## D-016 — No sector cap (Phase 7)
NSE archives provide no point-in-time sector classification (the current industry list
would leak today's classification into the past), so the 30% sector cap is not applied.

## D-017 — LSTM-AE compute budget (Phase 8)
On the development laptop (CPU only, PyTorch 2.12) one 512-sequence batch of the 2x64 LSTM-AE
takes ~0.36 s (8 threads was fastest; 2/4/16 threads were slower). To keep the anomaly stage
under ~1 hour across 9 yearly folds, each fold trains on a random sample of 60k sequences for
at most 10 epochs (patience 2). Chosen for compute, before seeing any results.

## D-018 — NSE `PREVCLOSE` is not adjusted; inference disabled (Phase 8, supersedes D-006)
Measured on the real data: for the 764 listed splits/bonuses, `prev_close / close(t-1)` matched
the event factor on **~0%** of ex-dates in every year 2014–2026 — NSE reports the raw previous
close. Inferring events from it produced 2,417 spurious "splits" (99.7% after a trading gap).
Inference is therefore off by default (`clean.infer_missing_ca_from_prevclose: false`; when
enabled it now requires consecutive trading days). Instead:
* the split parser was made robust to NSE's many phrasings ("Fv Splt Frm Rs 10 To Re 1",
  "Face Value Split Rs.10/- To Re.1/-"), recovering e.g. SBIN 2014 and JSWSTEEL 2017;
  only bonus debentures / preference shares / NCRPS remain unparsed, correctly;
* split-like jumps without an event stay `possible_missing_corporate_action` and are excluded;
* a listed event that does not explain the move (adjusted |return| > 30%) is `ca_price_mismatch`
  and excluded (replaces the meaningless `ca_prevclose_mismatch`).

## D-019 — Equity shares only; demergers are a known limitation (Phase 8)
ETFs (e.g. NIFTYBEES, BANKBEES) and other fund units trade in the EQ series. The panel keeps only
equity ISINs (`INE...`); fund units (`INF...`) are dropped. Demergers (e.g. Tata Motors 2025,
Siemens 2025, Piramal 2022) are not value-adjusted — that needs the spun-off company's price —
so the ex-date row is flagged as an unexplained jump and excluded; windows spanning it remain
distorted for the affected stock. About 10 liquid cases in 12 years.

## D-020 — Conclusion: does the signal exist? (Phase 5 research + Phase 8 final test)
Run `20261030-001505-ea0626`; development 2015–2024 (OOS scores from 2018, ranker OOS from 2020),
held-out test 2025-01-03 → 2026-10-29, evaluated once (`runs/TEST_LOG.jsonl`).

1. **Delivery-based accumulation is a real but small signal.** The one-line delivery-% z-score
   has a 5-day rank IC of 0.031 (95% CI 0.027–0.036) in development and 0.029 (0.023–0.035) in
   the test period. Event study: +0.1–0.24% excess return vs matched controls over 5 days,
   fading by 20–40 days.
2. **The deep anomaly model adds nothing over a one-line rule.** Raw LSTM-AE / Isolation-Forest
   anomaly percentiles have *negative* IC (−0.016 / −0.022 dev): unusual behaviour on its own
   predicts underperformance. Only after the accumulation (direction) filter do they turn
   positive (≈0.017), still below the delivery-z rule. In the ranker, adding anomaly scores
   changes 5-day IC by +0.001 (dev) and −0.003 (test) — noise.
3. **Delivery and bulk-deal features improve the ranker.** 5-day IC base → +delivery: 0.043 →
   0.052 (dev; CIs overlap), 0.058 → 0.069 (test); CPCV agrees (0.054 → 0.060). Gross backtest
   CAGR rises by ~11 points (dev) and ~10 points (test).
4. **Costs consume the edge.** A weekly, 5-day-horizon strategy turns the book over 35–42x a
   year; STT, stamp duty, charges and slippage cost ~17% a year. Net, the best variants roughly
   match the Nifty 500 in development (Sharpe 0.64–0.69 vs 0.65) and beat it by ~2–3 points in
   the test period (≈0% vs −2.2%), while trailing or tying the cost-free equal-weight universe.

Answer to the research question: India-specific delivery data carries information about
1–4-week excess returns that survives out of sample, but at this horizon and turnover it does
not survive realistic Indian trading costs, and the fraud-style deep anomaly detector adds
nothing beyond a simple delivery rule. Natural next steps (not tuned here, to keep the test
clean): a longer target horizon or monthly rebalancing to cut turnover, and a test of the
delivery rule as a filter on a low-turnover base strategy.
