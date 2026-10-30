"""Every DQ rule flags a constructed bad row and passes a good one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smd.clean import dq_checks as dq

GOOD = dict(open=100.0, high=105.0, low=98.0, close=103.0, volume=1000.0, delivery_pct=40.0,
            deliverable_qty=400.0, traded_qty=1000.0)


@pytest.mark.parametrize(
    "flag,bad",
    [
        ("high_lt_open_close", dict(high=102.0)),
        ("low_gt_open_close", dict(low=101.0)),
        ("negative_volume", dict(volume=-5.0)),
        ("delivery_pct_out_of_range", dict(delivery_pct=120.0)),
        ("deliverable_gt_traded", dict(deliverable_qty=1500.0)),
    ],
)
def test_rule_flags(flag, bad):
    df = pd.DataFrame([GOOD, {**GOOD, **bad}])
    out = dq.rule_checks(df)
    assert out[flag].tolist() == [False, True]
    assert not out.iloc[0].any()


def _panel(closes, split_event=None, volumes=None):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "isin": "X", "date": pd.bdate_range("2020-01-01", periods=n), "close": closes, "adj_close": closes,
        "split_event": split_event if split_event is not None else np.ones(n),
        "adj_volume": volumes if volumes is not None else np.full(n, 1000.0),
    })


def test_large_unexplained_move_flagged():
    out = dq.stat_checks(_panel([100, 101, 140, 141]), 0.30, 8, 60, 0.03)
    assert out["large_move_no_ca"].tolist() == [False, False, True, False]


def test_move_with_ca_not_flagged_as_missing():
    p = _panel([100, 101, 140, 141], split_event=np.array([1, 1, 0.5, 1.0]))
    out = dq.stat_checks(p, 0.30, 8, 60, 0.03)
    assert not out["large_move_no_ca"].any() and not out["possible_missing_corporate_action"].any()
    # the listed event does not explain the adjusted move -> ca_price_mismatch
    assert out["ca_price_mismatch"].tolist() == [False, False, True, False]


def test_split_like_jump_flagged_as_missing_ca():
    p = _panel([100, 100, 50.2, 50.5])  # looks like a 2:1 split / 1:1 bonus
    out = dq.stat_checks(p, 0.30, 8, 60, 0.03)
    assert out["possible_missing_corporate_action"].tolist() == [False, False, True, False]
    assert not out["large_move_no_ca"].any()


def test_volume_spike():
    rng = np.random.default_rng(0)
    vols = np.exp(rng.normal(np.log(1e5), 0.2, 80))
    vols[75] = 1e5 * 1e4
    out = dq.stat_checks(_panel(np.full(80, 100.0), volumes=vols), 0.30, 8, 60, 0.03)
    assert out["volume_spike"].sum() == 1 and out["volume_spike"].iloc[75]


def test_volume_spike_uses_only_past_rows():
    rng = np.random.default_rng(1)
    vols = np.exp(rng.normal(np.log(1e5), 0.2, 80))
    z1 = dq.robust_volume_z(_panel(np.full(80, 100.0), volumes=vols), 60)
    vols2 = vols.copy()
    vols2[70:] *= 50  # change the future
    z2 = dq.robust_volume_z(_panel(np.full(80, 100.0), volumes=vols2), 60)
    pd.testing.assert_series_equal(z1.iloc[:70], z2.iloc[:70])


def test_fixture_flags_planted_bad_rows(fixture_panel):
    flagged = fixture_panel[fixture_panel["dq_exclude"]]
    assert set(flagged["symbol"]) == {"STK05", "STK09", "STK11"}
    stk11 = fixture_panel[fixture_panel["symbol"] == "STK11"]
    assert stk11["dq_possible_missing_corporate_action"].sum() == 1
    assert fixture_panel["dq_ca_inferred_from_prevclose"].sum() == 0
