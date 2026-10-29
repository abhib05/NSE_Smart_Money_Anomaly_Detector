from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smd.config import UniverseCfg
from smd.features.base import trailing_z
from smd.features.registry import REGISTRY, feature_groups, normalize_cross_section
from smd.universe.point_in_time import compute_universe


def _panel(n_days=40, n_stocks=6, missing=None, start=None):
    days = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for k in range(n_stocks):
        for i, d in enumerate(days):
            if missing and (k, i) in missing:
                continue
            if start and i < start.get(k, 0):
                continue
            rows.append(dict(date=d, isin=f"S{k}", traded_value=1e6 * (k + 1), dq_exclude=False))
    return pd.DataFrame(rows), days


def test_universe_ranks_by_liquidity_and_caps_size():
    p, days = _panel()
    u = compute_universe(p, days, UniverseCfg(size=3, liquidity_lookback_days=10, min_history_days=10))
    last = u[u["date"] == days[-1]].set_index("isin")
    assert set(last[last["in_universe"]].index) == {"S5", "S4", "S3"}
    assert last.loc["S5", "liquidity_rank"] == 1
    sizes = u[u["in_universe"]].groupby("date").size()
    assert (sizes == 3).all()


def test_min_history_rule():
    p, days = _panel(start={5: 30})  # most liquid stock lists late
    u = compute_universe(p, days, UniverseCfg(size=3, liquidity_lookback_days=5, min_history_days=8))
    on = u[(u["isin"] == "S5") & u["in_universe"]]["date"]
    assert on.min() == days[30 + 7]


def test_missing_days_rule():
    missing = {(5, i) for i in range(20, 25)}
    p, days = _panel(missing=missing)
    u = compute_universe(p, days, UniverseCfg(size=3, liquidity_lookback_days=10, min_history_days=10, max_missing_frac=0.05))
    s5 = u[(u["isin"] == "S5") & (u["date"] > days[25]) & (u["date"] <= days[33])]
    assert s5.empty  # >5% of the last 10 days missing


def test_delisted_stock_stays_in_history(fixture_cfg, fixture_panel):
    from smd.clean.panel import trading_days

    days = trading_days(fixture_cfg)
    u = compute_universe(fixture_panel, days, fixture_cfg.universe)
    sid = fixture_panel.loc[fixture_panel["symbol"] == "STK20", "isin"].iloc[0]
    hist = u[(u["isin"] == sid)]
    assert hist["date"].max() < days[-100]  # stops when it stops trading
    assert len(hist) > 100  # but present before


def test_trailing_z_excludes_today():
    x = pd.DataFrame({"a": [1.0, 2, 3, 4, 100]})
    z = trailing_z(x, 4, 4)
    prev = np.array([1.0, 2, 3, 4])
    assert z.iloc[4, 0] == pytest.approx((100 - prev.mean()) / prev.std(ddof=1))
    assert np.isnan(z.iloc[3, 0])


def test_normalize_cross_section_range_and_order():
    df = pd.DataFrame({"date": ["d1"] * 5 + ["d2"] * 3, "x": [1, 2, 3, 4, 1000, 5, np.nan, 6]})
    out = normalize_cross_section(df, ["x"], winsor=0.0)
    assert out["x"].between(-1, 1).all()
    assert list(out["x"][:5]) == sorted(out["x"][:5])
    assert out["x"].iloc[0] == pytest.approx(-1) and out["x"].iloc[4] == pytest.approx(1)
    assert out["x"].iloc[6] == 0.0  # missing -> neutral


def test_registry_metadata():
    groups = feature_groups()
    assert {"price", "liquidity", "delivery", "bulk"} <= set(groups)
    for f in REGISTRY.values():
        assert f.lookback >= 1 and f.description


def test_fixture_universe_size_stable(fixture_cfg, fixture_panel):
    from smd.clean.panel import trading_days

    days = trading_days(fixture_cfg)
    u = compute_universe(fixture_panel, days, fixture_cfg.universe)
    sizes = u[u["in_universe"]].groupby("date").size()
    study = sizes[sizes.index >= pd.Timestamp(fixture_cfg.study_start)]
    assert (study == fixture_cfg.universe.size).mean() > 0.95
