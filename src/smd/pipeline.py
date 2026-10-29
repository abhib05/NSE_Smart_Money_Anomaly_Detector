"""Stage orchestration: each function reads its inputs from the processed store, writes its
outputs, validates them, and returns summary metrics for the run manifest."""

from __future__ import annotations

import pandas as pd

from smd.clean.panel import trading_days
from smd.config import Config
from smd.utils import get_logger, read_dataset, write_dataset
from smd.validation.schemas import validate

log = get_logger(__name__)

PANEL_FEATURE_COLUMNS = [
    "date", "isin", "symbol", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "traded_value",
    "adj_deliverable_qty", "delivery_pct", "dq_exclude",
]


def run_universe(cfg: Config) -> dict:
    from smd.universe.point_in_time import compute_universe, universe_size_by_date

    days = trading_days(cfg)
    panel = read_dataset(cfg, "silver", "daily_panel", columns=["date", "isin", "traded_value", "dq_exclude"])
    uni = validate(compute_universe(panel, days, cfg.universe), "gold.universe")
    write_dataset(cfg, "gold", "universe", uni)
    sizes = universe_size_by_date(uni)
    study = sizes[sizes.index >= pd.Timestamp(cfg.study_start)]
    return {
        "rows": len(uni),
        "securities_ever_in_universe": int(uni.loc[uni["in_universe"], "isin"].nunique()),
        "size_min_in_study": int(study.min()) if len(study) else 0,
        "size_median_in_study": float(study.median()) if len(study) else 0.0,
        "first_full_date": str(sizes[sizes >= cfg.universe.size].index.min()),
    }


def run_features(cfg: Config) -> dict:
    from smd.features.registry import build_features

    days = trading_days(cfg)
    panel = read_dataset(cfg, "silver", "daily_panel", columns=PANEL_FEATURE_COLUMNS)
    deals = read_dataset(cfg, "silver", "deals")
    uni = read_dataset(cfg, "gold", "universe")
    feats, beh = build_features(panel, deals, uni, days, cfg)
    feats = validate(feats, "gold.features")
    write_dataset(cfg, "gold", "features", feats)
    write_dataset(cfg, "gold", "behaviour", beh)
    cov = feats.drop(columns=["date", "isin"]).notna().mean().round(3).to_dict()
    return {"rows": len(feats), "behaviour_rows": len(beh), "coverage": cov}


def run_labels(cfg: Config) -> dict:
    from smd.labels.forward_returns import build_labels

    days = trading_days(cfg)
    panel = read_dataset(cfg, "silver", "daily_panel", columns=["date", "isin", "adj_open_tr", "adj_close_tr", "dq_exclude"])
    uni = read_dataset(cfg, "gold", "universe")
    labels = validate(build_labels(panel, uni, days, cfg.labels.horizons, cfg.labels.execution), "gold.labels")
    write_dataset(cfg, "gold", "labels", labels)
    return {
        "rows": len(labels),
        **{f"mean_fwd_ret_{h}d": float(labels[f"fwd_ret_{h}d"].mean()) for h in cfg.labels.horizons},
    }
