"""Ranker tests, including the CV leakage test on the rows the ranker actually trains on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smd.cv.splits import train_mask
from smd.model.ranker import (
    ANOMALY_FEATURES,
    VARIANTS,
    ablation_table,
    load_model_frame,
    ranker_folds,
    rebalance_dates,
    train_predict,
    variant_features,
)


@pytest.fixture(scope="module")
def model_frame(fixture_cfg, fixture_scores):
    from smd.clean.panel import trading_days

    days = trading_days(fixture_cfg)
    return load_model_frame(fixture_cfg, days), days


def test_variants_are_nested():
    a, b, c = (set(variant_features(v)) for v in VARIANTS)
    assert a < b < c
    assert set(ANOMALY_FEATURES) <= c and not (set(ANOMALY_FEATURES) & b)
    assert "delivery_pct_z_60d" in b and "delivery_pct_z_60d" not in a


def test_rebalance_dates_are_last_trading_day_of_week():
    days = pd.bdate_range("2024-01-01", "2024-01-31").delete(4)  # Friday 5th is a holiday
    rb = rebalance_dates(days)
    assert pd.Timestamp("2024-01-04") in rb and pd.Timestamp("2024-01-12") in rb


def test_cv_leakage_on_ranker_rows(fixture_cfg, model_frame):
    df, days = model_frame
    folds = ranker_folds(fixture_cfg, df, days)
    assert len(folds) >= 2
    for fold in folds:
        tr = train_mask(df["date"], df["label_end_5d"], fold, days, fixture_cfg.cv.embargo_days)
        te = (df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)
        assert (pd.to_datetime(df.loc[tr, "label_end_5d"]) < fold.test_start).all()
        gap = days.searchsorted(fold.test_start) - days.get_indexer(df.loc[tr, "date"]).max()
        assert gap >= fixture_cfg.cv.embargo_days
        assert not (tr & te).any()


def test_predictions_out_of_sample_and_deterministic(fixture_cfg, model_frame):
    df, days = model_frame
    p1, _, info = train_predict(fixture_cfg, df, days, ("base",))
    p2, _, _ = train_predict(fixture_cfg, df, days, ("base",))
    np.testing.assert_array_equal(p1["score"].to_numpy(), p2["score"].to_numpy())
    folds = {f.fold_id: f for f in ranker_folds(fixture_cfg, df, days)}
    for fid, g in p1.groupby("fold_id"):
        assert g["date"].min() >= folds[fid].test_start
    assert all(i["n_train"] > 0 for i in info)


def test_ablation_table_dev_only(fixture_cfg, model_frame):
    df, days = model_frame
    preds, _, _ = train_predict(fixture_cfg, df, days)
    t = ablation_table(fixture_cfg, preds, df, "dev")
    assert list(t["variant"]) == list(VARIANTS)
    assert (t["period"] == "dev").all()
    n_dev_dates = preds[preds["date"] <= pd.Timestamp(fixture_cfg.dev_end)]["date"].nunique()
    assert (t["n_dates_5d"] <= n_dev_dates).all()
