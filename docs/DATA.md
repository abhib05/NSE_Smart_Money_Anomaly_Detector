# Data

This file records what data the project uses, where it comes from, how far back it
goes, and what the data-quality (DQ) checks found. Sections marked *(generated)* are
refreshed by the pipeline (`smd clean` writes `reports/dq_summary.md`, which is
copied here after a full run).

## 1. Phase 0 — availability spike (probed 2026-10-29)

Raw probe output: [`phase0_spike_results.md`](phase0_spike_results.md)
(re-run with `python scripts/phase0_spike.py`).

| Dataset | Source used | Coverage verified | Schema versions | Verdict |
|---|---|---|---|---|
| Equity bhavcopy (OHLCV, ISIN) | `nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{MMM}/cm{dd}{MMM}{yyyy}bhav.csv.zip` | 2012 → 2024-07-05 | **v1 "old"** CSV (`SYMBOL,SERIES,OPEN,…,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN`) | usable |
| Equity bhavcopy (UDiFF) | `nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip` | 2024-03 (overlap) → 2026-10 | **v2 "UDiFF"** CSV (`TradDt,…,ISIN,TckrSymb,SctySrs,…,OpnPric,…,TtlTradgVol,TtlTrfVal,…`) | usable |
| Security-wise delivery (MTO) | `nsearchives.nseindia.com/archives/equities/mto/MTO_{ddmmyyyy}.DAT` | 2012 → 2026-10 | one format (4 header lines + `20,sr,symbol,series,traded,deliverable,pct`) | **usable — this is the critical dataset** |
| Full bhavcopy with delivery | `…/products/content/sec_bhavdata_full_{ddmmyyyy}.csv` (what `aynse.full_bhavcopy_*` uses) | only ~2020 → | one format | **not used** — too short; MTO covers the same delivery fields back to 2012 |
| Bulk & block deals | `nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals|block_deals&from=…&to=…&csv=true` | 2012 → 2026-10 | CSV (`Date,Symbol,Security Name,Client Name,Buy / Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks`) | usable — **must use `csv=true`**, JSON truncates at 70 rows |
| Index snapshot (Nifty 500 etc.) | `niftyindices.com/Daily_Snapshot/ind_close_all_{ddmmyyyy}.csv` | 2012 → 2026-10 | one format | usable (price index only; no TRI in the snapshot) |
| Corporate actions | `nseindia.com/api/corporates-corporateActions?index=equities&from_date=…&to_date=…` | 2012 → 2026 (≈1.8k–2.7k rows/yr) | JSON (`symbol,series,isin,exDate,subject,faceVal,…`) | usable — free-text `subject` must be parsed |
| Symbol changes | `nsearchives.nseindia.com/content/equities/symbolchange.csv` | snapshot | one format | used as a cross-check only |

### Observations that shaped the design
* **Bhavcopy schema switch.** NSE moved the equity bhavcopy to the UDiFF format; the
  old URL stops serving after 2024-07-05 and UDiFF files exist from ~2024-03. The loader
  uses the old file for dates `< 2024-07-08` and UDiFF from `2024-07-08`, and both
  parsers normalise into one canonical schema (unit-tested with real snippets).
* **`aynse` is used only as a reference.** Its post-2024 bhavcopy path returned the
  *corporate-actions* file instead of prices during the spike, its `full_bhavcopy`
  only reaches 2020, and it parses/coerces data before it can be stored. To keep
  `data/raw/` byte-for-byte immutable, ingestion downloads the archive files directly
  with a throttled `httpx` client (the same URLs `aynse` uses).
* **ISIN is not perfectly stable**: on a face-value split NSE usually issues a new
  ISIN (e.g. Greenply `INE461C01012 → INE461C01020`). The cleaner chains ISINs into a
  stable security id (see DECISIONS.md D-004).
* **Trading calendar.** There is no reliable historical holiday file. A weekday with no
  bhavcopy is treated as a holiday; a weekday where the index snapshot exists but
  bhavcopy/delivery is missing is logged as a **gap**. Weekend special sessions
  (Muhurat, Budget-day Saturdays) are not downloaded — a documented limitation.

## 2. Study-period decision

All critical datasets (bhavcopy, delivery, deals, corporate actions, index) are
available from **2012**, so nothing needs to be dropped to a sub-period.

| Setting | Value | Why |
|---|---|---|
| Ingest start | 2014-01-01 | one year of warm-up before the study start (252-day features, 200-day history rule) |
| Study start | 2015-01-01 | ~10 years of usable history |
| Development period | 2015-01-01 → 2024-12-31 | all tuning, research and ablation |
| Held-out test period | 2025-01-01 → latest (2026-10) | touched once in Phase 8 |

The test period starts on a calendar-year boundary because walk-forward folds retrain
every 12 months on calendar years (skfolio `WalkForward(freq="12MS")`); this keeps every
fold entirely inside either the development or the test period.

## 3. Ingestion statistics (full ingest 2014-01-01 → 2026-10-28, run 2026-10-29)

| Dataset | Files OK | 404 / not published | Errors | Notes |
|---|---:|---:|---:|---|
| bhavcopy (old CSV + UDiFF) | 3,161 | 185 | 0 | 185 weekday 404s = exchange holidays |
| index snapshots | 3,076 | 270 | 0 | niftyindices misses 85 trading days (not a gap in our data) |
| delivery (MTO) | 3,161 | 0 | 0 | one file per trading day — no gaps |
| bulk deals (monthly CSV) | 154 | 0 | 0 | |
| block deals (monthly CSV) | 154 | 0 | 0 | |
| corporate actions (quarterly) | 53 | 0 | 0 | 28,231 announcements |

Raw store: 574 MB. Re-running `smd ingest` immediately afterwards requested **0** files
(idempotency check, Phase 1 acceptance). Trading-calendar gaps (trading day with bhavcopy but
no delivery file, or index but no bhavcopy): **0**.

Bronze: 5.79M bhavcopy rows (EQ + BE), 5.30M delivery rows, 173k deal rows, 276k index rows.

Real-file quirks found and handled (each has a parser regression test):
* niftyindices writes `dd/mm/yyyy` instead of `dd-mm-yyyy` in parts of 2015–2016, and has rows
  with a closing value of 0 (dropped);
* 9 MTO files (2017–2019) garble the header label as `rade Date <…>`;
* one bhavcopy (2020-07-13) uses a 2-digit year (`13-Jul-20`).

## 4. Cleaning and data-quality flags

Silver panel: **4,913,597 rows, 3,160 equity securities, 3,161 trading days** (EQ series,
equity ISINs only — ETFs excluded, D-019). 780 split/bonus/consolidation events and 15,763
dividends applied; 479 ISIN changes linked into stable security ids.

| flag | rows | % of rows | excluded from features |
|---|---:|---:|---|
| high_lt_open_close | 0 | 0.0000 | yes |
| low_gt_open_close | 0 | 0.0000 | yes |
| negative_volume | 0 | 0.0000 | yes |
| delivery_pct_out_of_range | 0 | 0.0000 | yes |
| deliverable_gt_traded | 0 | 0.0000 | yes |
| possible_missing_corporate_action | 1,458 | 0.0297 | yes |
| large_move_no_ca | 1,354 | 0.0276 | yes |
| ca_price_mismatch | 3 | 0.0001 | yes |
| volume_spike | 1,163 | 0.0237 | no |
| ca_inferred_from_prevclose | 0 | 0.0000 | no |
| div_too_large | 3 | 0.0001 | no |
| ANY EXCLUDED | 2,815 | 0.0573 | yes |

`possible_missing_corporate_action` / `large_move_no_ca` are dominated by illiquid stocks and
returns after trading gaps; among liquid names they are real crashes (YESBANK 2020, DHFL 2018,
Jet Airways 2019) and demergers (Tata Motors 2025, Siemens 2025), see D-019.

### Corporate-action spot-check (Phase 2 acceptance)

`python scripts/spotcheck_corporate_actions.py --n 6` — the most liquid securities with a
split/bonus in the study period. Raw ex-date moves of -50% to -90% become ordinary daily moves
after adjustment:

| symbol | ex_date | event | raw_return_on_ex_date | adjusted_return_on_ex_date |
|---|---|---|---|---|
| RELIANCE | 2017-09-07 | bonus:0.5 | -50.3% | -0.56% |
| ICICIBANK | 2017-06-20 | bonus:0.9091;dividend:2.50 | -8.8% | +0.34% |
| HDFCBANK | 2025-08-26 | bonus:0.5 | -50.4% | -0.88% |
| INFY | 2015-06-15 | bonus:0.5;dividend:29.50 | -49.9% | +0.30% |
| TCS | 2018-05-31 | bonus:0.5;dividend:29.00 | -50.5% | -0.91% |
| BAJFINANCE | 2016-09-08 | bonus:0.5;split:0.2 | -89.8% | +2.06% |

## 5. Latest pipeline run

Run `20261030-001505-ea0626` (data through 2026-10-29): universe exactly 500 stocks on every
study day (1,321 distinct securities over time); 1.48M feature rows; 1.09M out-of-sample anomaly
scores per model from 2018-01-01; ranker predictions from 2020 on 528k weekly rows. Full results:
[`reports/latest/REPORT.md`](../reports/latest/REPORT.md); conclusion in DECISIONS.md D-020.
