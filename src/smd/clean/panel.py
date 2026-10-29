"""Bronze -> silver: the clean daily panel (one row per security per trading day).

Steps: EQ series only -> join delivery -> stable security id (isin_map) -> corporate-action
events (listed + inferred) -> backward adjustment and total-return factors -> DQ checks.
Also writes silver `deals` (mapped to security ids), `corporate_actions`, `isin_map` and
`benchmark` (Nifty 500 index).
"""

from __future__ import annotations

import pandas as pd

from smd.clean import corporate_actions as cax
from smd.clean.dq_checks import dq_summary, run_dq
from smd.clean.isin_map import build_isin_map, map_symbol_date_to_sid
from smd.config import Config
from smd.utils import get_logger, read_dataset, write_dataset
from smd.validation.schemas import validate

log = get_logger(__name__)

PANEL_COLUMNS = [
    "date", "isin", "raw_isin", "symbol", "open", "high", "low", "close", "prev_close",
    "adj_factor", "tr_factor", "adj_open", "adj_high", "adj_low", "adj_close", "adj_open_tr", "adj_close_tr",
    "volume", "adj_volume", "traded_value", "trades", "traded_qty", "deliverable_qty", "adj_deliverable_qty",
    "delivery_pct", "ca_event", "dq_flags", "dq_exclude",
]
BENCHMARK_INDEX = "Nifty 500"


def trading_days(cfg: Config) -> pd.DatetimeIndex:
    cal = read_dataset(cfg, "bronze", "calendar")
    return pd.DatetimeIndex(sorted(cal.loc[cal["is_trading_day"], "date"]))


def build_panel(bhav: pd.DataFrame, delivery: pd.DataFrame, ca_raw: pd.DataFrame, days: pd.DatetimeIndex,
                cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eq = bhav[bhav["series"] == "EQ"].drop(columns=["source_file"], errors="ignore")
    dl = delivery[delivery["series"] == "EQ"].drop(columns=["source_file"], errors="ignore")
    p = eq.merge(dl, on=["date", "symbol", "series"], how="left")

    imap = build_isin_map(eq, days)
    p["raw_isin"] = p["isin"].fillna("SYM:" + p["symbol"])
    isin_to_sid = imap.drop_duplicates("isin").set_index("isin")["sid"]
    p["isin"] = p["raw_isin"].map(isin_to_sid)
    before = len(p)
    p = p.sort_values(["isin", "date", "volume"]).drop_duplicates(["isin", "date"], keep="last")
    if len(p) < before:
        log.warning("dropped %d duplicate (sid, date) rows", before - len(p))
    p = p.sort_values(["isin", "date"]).reset_index(drop=True)

    ca = cax.parse_corporate_actions(ca_raw)
    sid_by_isin = ca["isin"].map(isin_to_sid)
    sid_by_symbol = map_symbol_date_to_sid(ca.rename(columns={"ex_date": "date"}), imap, max_staleness_days=30)
    ca["sid"] = sid_by_isin.fillna(sid_by_symbol)
    log.info("corporate actions: %d parsed events, %d mapped to a security", len(ca), int(ca["sid"].notna().sum()))

    p = cax.event_factors(p, ca, cfg.clean.infer_missing_ca_from_prevclose, cfg.clean.split_ratio_tolerance)
    p = cax.adjustment_factors(p)
    p = run_dq(p, cfg.clean.max_abs_return, cfg.clean.volume_z_threshold, cfg.clean.volume_z_lookback,
               cfg.clean.split_ratio_tolerance)
    dq_cols = [c for c in p.columns if c.startswith("dq_") and c not in ("dq_flags", "dq_exclude")]
    panel = p[PANEL_COLUMNS + dq_cols].sort_values(["date", "isin"]).reset_index(drop=True)
    return panel, imap, ca


def build_silver(cfg: Config) -> dict:
    days = trading_days(cfg)
    bhav = read_dataset(cfg, "bronze", "bhavcopy")
    delivery = read_dataset(cfg, "bronze", "delivery")
    ca_raw = read_dataset(cfg, "bronze", "corporate_actions")
    panel, imap, ca = build_panel(bhav, delivery, ca_raw, days, cfg)
    validate(panel, "silver.daily_panel")

    deals = read_dataset(cfg, "bronze", "deals")
    deals = deals.assign(isin=map_symbol_date_to_sid(deals, imap)).dropna(subset=["isin"])
    deals["value"] = deals["qty"] * deals["price"]
    deals = validate(deals.drop(columns=["source_file"]).reset_index(drop=True), "silver.deals")

    idx = read_dataset(cfg, "bronze", "index")
    bench = idx[idx["index_name"].str.lower() == BENCHMARK_INDEX.lower()][["date", "close"]]
    bench = bench.rename(columns={"close": "nifty500"}).sort_values("date").reset_index(drop=True)

    write_dataset(cfg, "silver", "daily_panel", panel)
    write_dataset(cfg, "silver", "isin_map", imap)
    write_dataset(cfg, "silver", "corporate_actions", ca)
    write_dataset(cfg, "silver", "deals", deals)
    write_dataset(cfg, "silver", "benchmark", bench)
    summary = dq_summary(panel)
    write_dataset(cfg, "silver", "dq_summary", summary)
    write_dq_report(cfg, panel, summary, ca)
    return {
        "panel_rows": len(panel),
        "securities": int(panel["isin"].nunique()),
        "dates": int(panel["date"].nunique()),
        "excluded_rows": int(panel["dq_exclude"].sum()),
        "deals_rows": len(deals),
        "ca_events": len(ca),
    }


def write_dq_report(cfg: Config, panel: pd.DataFrame, summary: pd.DataFrame, ca: pd.DataFrame) -> None:
    lines = [
        "# Data-quality summary",
        "",
        f"Panel: {len(panel):,} rows, {panel['isin'].nunique():,} securities, "
        f"{panel['date'].min():%Y-%m-%d} -> {panel['date'].max():%Y-%m-%d}.",
        "",
        "| flag | rows | % of rows | excluded from features |",
        "|---|---:|---:|---|",
    ]
    for r in summary.itertuples():
        lines.append(f"| {r.flag} | {r.rows:,} | {r.pct:.4f} | {'yes' if r.excluded else 'no'} |")
    counts = ca["action_type"].value_counts()
    lines += ["", "Corporate-action events parsed: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()), ""]
    out = cfg.paths.reports_dir / "dq_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def spot_check(cfg: Config, symbol: str, around: str, window: int = 3) -> pd.DataFrame:
    """Raw vs adjusted prices for `symbol` around a date - for manual corporate-action checks."""
    panel = read_dataset(cfg, "silver", "daily_panel",
                         columns=["date", "isin", "symbol", "close", "prev_close", "adj_factor", "adj_close",
                                  "adj_close_tr", "volume", "adj_volume", "ca_event", "dq_flags"])
    sid = panel.loc[panel["symbol"] == symbol, "isin"]
    if sid.empty:
        raise KeyError(symbol)
    s = panel[panel["isin"].isin(sid.unique())].sort_values("date").reset_index(drop=True)
    i = int(s["date"].searchsorted(pd.Timestamp(around)))
    return s.iloc[max(0, i - window): i + window + 1]
