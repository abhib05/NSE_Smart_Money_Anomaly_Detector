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

## 3. Ingestion statistics *(generated)*

_Filled in after the full ingest — see section 5._

## 4. Data-quality flags *(generated)*

_Filled in after `smd clean` — see section 5._

## 5. Latest pipeline run *(generated)*

_Pending first full run._
