"""Corporate-action parsing and adjustment: synthetic splits/bonuses must give continuous prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smd.clean import corporate_actions as cax


@pytest.mark.parametrize(
    "subject,kind,factor",
    [
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", "split", 0.2),
        (" Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share", "split", 0.2),
        ("Bonus 1:1", "bonus", 0.5),
        (" Bonus 1:2", "bonus", 2 / 3),
        ("Bonus 3:1", "bonus", 0.25),
        ("Consolidation Of Shares From Re 1/- To Rs 10/-", "consolidation", 10.0),
    ],
)
def test_parse_price_events(subject, kind, factor):
    ev = cax.parse_subject(subject)
    assert ev[0]["action_type"] == kind
    assert ev[0]["price_factor"] == pytest.approx(factor)


@pytest.mark.parametrize(
    "subject,cash",
    [
        ("Interim Dividend - Rs 5.50/- Per Share", 5.5),
        ("Final Dividend Rs 6/- Plus Special Dividend Rs 4/- Per Share", 10.0),
        ("Interim Dividend  - Re 0.20/- Per Share", 0.2),
    ],
)
def test_parse_dividends(subject, cash):
    ev = cax.parse_subject(subject)
    assert ev[0]["action_type"] == "dividend" and ev[0]["cash_amount"] == pytest.approx(cash)


def test_percentage_dividend_uses_face_value():
    ev = cax.parse_subject("Dividend - 50%", face_value=10)
    assert ev[0]["cash_amount"] == pytest.approx(5.0)


def test_irrelevant_subjects_produce_no_events():
    assert cax.parse_subject("Annual General Meeting") == []


def test_rights_recorded_without_adjustment():
    ev = cax.parse_subject("Rights 1:5 @ Premium Rs 90/-")
    assert ev[0]["action_type"] == "rights" and ev[0]["price_factor"] == 1.0


def _series(raw_close, prev_close, events=None, divs=None):
    n = len(raw_close)
    p = pd.DataFrame({
        "isin": "X", "date": pd.bdate_range("2020-01-01", periods=n), "close": raw_close, "prev_close": prev_close,
        "open": raw_close, "high": raw_close, "low": raw_close, "volume": 1000.0, "deliverable_qty": 500.0,
    })
    ca = pd.DataFrame(events or [], columns=["sid", "ex_date", "action_type", "price_factor", "cash_amount"])
    for i, cash in (divs or {}).items():
        ca.loc[len(ca)] = ["X", p["date"][i], "dividend", 1.0, cash]
    return p, ca


def test_split_gives_continuous_adjusted_prices():
    raw = np.array([100.0, 102, 104, 20.8 * 1.01, 21.0])  # 5:1 split on day 3
    prev = np.array([np.nan, 100, 102, 104 * 0.2, 20.8 * 1.01])
    p, ca = _series(raw, prev, [["X", pd.Timestamp("2020-01-06"), "split", 0.2, 0.0]])
    out = cax.adjustment_factors(cax.event_factors(p, ca, infer_from_prevclose=False, tol=0.03))
    assert out["adj_factor"].tolist() == pytest.approx([0.2, 0.2, 0.2, 1.0, 1.0])
    rets = out["adj_close"].pct_change().dropna()
    assert rets.abs().max() < 0.03
    assert out["adj_volume"].iloc[0] == pytest.approx(1000 / 0.2)  # volume scaled up before the split


def test_bonus_adjustment():
    raw = np.array([300.0, 303, 151.5, 152])  # 1:1 bonus on day 2
    p, ca = _series(raw, np.r_[np.nan, 300, 151.5, 151.5], [["X", pd.Timestamp("2020-01-03"), "bonus", 0.5, 0.0]])
    out = cax.adjustment_factors(cax.event_factors(p, ca, False, 0.03))
    assert out["adj_close"].iloc[1] == pytest.approx(151.5)
    assert out["adj_close"].iloc[2] == pytest.approx(151.5)


def test_missing_split_inferred_from_prevclose():
    raw = np.array([500.0, 505, 101.0, 102])
    prev = np.array([np.nan, 500, 101.0, 101])  # exchange-adjusted prev close on ex-date
    p, ca = _series(raw, prev)
    out = cax.event_factors(p, ca, infer_from_prevclose=True, tol=0.03)
    assert out["ca_inferred_from_prevclose"].tolist() == [False, False, True, False]
    adj = cax.adjustment_factors(out)
    assert adj["adj_close"].pct_change().abs().max() < 0.02


def test_no_inference_when_disabled():
    p, ca = _series(np.array([500.0, 505, 101.0]), np.array([np.nan, 500, 101.0]))
    out = cax.event_factors(p, ca, infer_from_prevclose=False, tol=0.03)
    assert not out["ca_inferred_from_prevclose"].any()


def test_dividend_total_return():
    raw = np.array([100.0, 100.0, 95.0, 95.0])  # Rs 5 dividend, ex on day 2
    p, ca = _series(raw, np.r_[np.nan, 100, 100, 95], divs={2: 5.0})
    out = cax.adjustment_factors(cax.event_factors(p, ca, False, 0.03))
    assert out["adj_close"].tolist() == pytest.approx(raw.tolist())  # price series untouched
    tr = out["adj_close_tr"].pct_change()
    assert tr.iloc[2] == pytest.approx(0.0, abs=1e-12)  # dividend added back
    assert out["tr_factor"].iloc[0] == pytest.approx(0.95)


def test_typical_factor_matching():
    assert cax.match_typical_factor(0.502, 0.03) == pytest.approx(0.5)
    assert cax.match_typical_factor(0.2, 0.03) == pytest.approx(0.2)
    assert cax.match_typical_factor(0.9, 0.03) is None


def test_fixture_split_isin_chain(fixture_panel):
    """Fixture STK03 has a 10->2 split with an ISIN change; adjusted TR series must be continuous."""
    p = fixture_panel.sort_values(["isin", "date"])
    sid = p.loc[p["symbol"] == "STK03", "isin"].iloc[0]
    s = p[p["isin"] == sid]
    assert s["raw_isin"].nunique() == 2
    assert s["adj_close_tr"].pct_change().abs().max() < 0.2
