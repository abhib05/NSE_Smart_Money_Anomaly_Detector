"""Look-ahead tests (the most important tests in the project).

Future-perturbation test: randomly change *all raw data after date t* (prices, volumes,
delivery, deals, plus a new corporate action), rebuild the whole pipeline from bronze, and
assert that every feature, the universe and the behaviour channels for dates <= t are
unchanged. Floating-point tolerance 1e-9 (relative) - see DECISIONS.md D-008.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smd.clean.panel import build_panel
from smd.config import load_config
from smd.features.registry import BEHAVIOUR_CHANNELS, REGISTRY, XS_FEATURES, build_features
from smd.fixture import make_fixture
from smd.labels.forward_returns import build_labels
from smd.universe.point_in_time import compute_universe

CFG = load_config("configs/fixture.yaml")


def _pipeline(tables):
    days = pd.DatetimeIndex(sorted(tables["calendar"]["date"]))
    panel, _, _ = build_panel(tables["bhavcopy"], tables["delivery"], tables["corporate_actions"], days, CFG)
    deals = tables["deals"].copy()
    from smd.clean.isin_map import build_isin_map, map_symbol_date_to_sid

    imap = build_isin_map(tables["bhavcopy"][tables["bhavcopy"]["series"] == "EQ"], days)
    deals["isin"] = map_symbol_date_to_sid(deals, imap)
    deals = deals.dropna(subset=["isin"])
    deals["value"] = deals["qty"] * deals["price"]
    uni = compute_universe(panel, days, CFG.universe)
    feats, beh = build_features(panel, deals, uni, days, CFG)
    labels = build_labels(panel, uni, days, CFG.labels.horizons, CFG.labels.execution)
    return dict(panel=panel, universe=uni, features=feats, behaviour=beh, labels=labels, days=days)


def _perturb_after(tables, t: pd.Timestamp, seed: int):
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in tables.items()}
    b = out["bhavcopy"]
    fut = b["date"] > t
    scale = rng.uniform(0.5, 1.6, fut.sum())
    for c in ("open", "high", "low", "close", "last", "prev_close"):
        b.loc[fut, c] = b.loc[fut, c].to_numpy() * scale
    b.loc[fut, "volume"] = np.round(b.loc[fut, "volume"].to_numpy() * rng.uniform(0.1, 20, fut.sum()))
    b.loc[fut, "traded_value"] = b.loc[fut, "traded_value"].to_numpy() * rng.uniform(0.1, 20, fut.sum())
    d = out["delivery"]
    fd = d["date"] > t
    d.loc[fd, "delivery_pct"] = rng.uniform(1, 100, fd.sum()).round(2)
    d.loc[fd, "deliverable_qty"] = np.floor(d.loc[fd, "traded_qty"] * d.loc[fd, "delivery_pct"] / 100)
    deals = out["deals"]
    future_days = sorted(b.loc[fut, "date"].unique())[:50]
    extra = pd.DataFrame({
        "date": rng.choice(future_days, 40), "symbol": rng.choice(b["symbol"].unique(), 40),
        "client_name": [f"FUTURE {i}" for i in range(40)], "side": rng.choice(["BUY", "SELL"], 40),
        "qty": 1e5, "price": 100.0, "deal_type": "bulk", "source_file": "perturbed",
    })
    out["deals"] = pd.concat([deals[deals["date"] <= t], extra], ignore_index=True)
    ca = out["corporate_actions"]
    split_day = future_days[10]
    sym = "STK01"
    new_ca = pd.DataFrame([{
        "symbol": sym, "series": "EQ", "isin": b.loc[b["symbol"] == sym, "isin"].iloc[0], "ex_date": split_day,
        "record_date": split_day, "subject": "Bonus 1:1", "face_value": 10.0, "source_file": "perturbed",
    }])
    out["corporate_actions"] = pd.concat([ca, new_ca], ignore_index=True)
    return out


@pytest.fixture(scope="module", params=[(330, 1), (520, 2)], ids=["cutoff330", "cutoff520"])
def runs(request):
    tables, _ = make_fixture(7)
    days = pd.DatetimeIndex(sorted(tables["calendar"]["date"]))
    t = days[request.param[0]]
    base = _pipeline(tables)
    pert = _pipeline(_perturb_after(tables, t, request.param[1]))
    return t, base, pert


def _past(df, t):
    return df[df["date"] <= t].set_index(["date", "isin"]).sort_index()


ALL_FEATURES = list(REGISTRY) + list(XS_FEATURES)


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_feature_has_no_lookahead(runs, feature):
    t, base, pert = runs
    a, b = _past(base["features"], t)[feature], _past(pert["features"], t)[feature]
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("channel", BEHAVIOUR_CHANNELS)
def test_behaviour_channel_has_no_lookahead(runs, channel):
    # The behaviour table covers every security that is *ever* in the universe, so which
    # securities have rows can depend on the future; the *values* must not. Sequences are
    # only built for (date, isin) pairs in the point-in-time universe (tested above).
    t, base, pert = runs
    a, b = _past(base["behaviour"], t)[channel], _past(pert["behaviour"], t)[channel]
    common = a.index.intersection(b.index)
    assert len(common) > 0.5 * min(len(a), len(b))
    a, b = a.loc[common], b.loc[common]
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_universe_has_no_lookahead(runs):
    t, base, pert = runs
    a, b = _past(base["universe"], t), _past(pert["universe"], t)
    pd.testing.assert_frame_equal(a, b)


def test_perturbation_actually_changed_the_future(runs):
    t, base, pert = runs
    fa = base["features"][base["features"]["date"] > t]["ret_5d"].to_numpy()
    fb = pert["features"][pert["features"]["date"] > t]["ret_5d"].to_numpy()
    assert len(fa) != len(fb) or not np.allclose(fa, fb, equal_nan=True)


# ---- label alignment ----------------------------------------------------------------------

def test_labels_use_only_prices_from_t_plus_1(runs):
    """Labels at t must change when t+1.. prices change and must not depend on prices <= t."""
    t, base, pert = runs
    lab_b, lab_p = base["labels"].set_index(["date", "isin"]), pert["labels"].set_index(["date", "isin"])
    days = base["days"]
    pos = days.get_loc(t)
    # labels dated well before t (window ends before t) are unchanged
    early = days[pos - 25]
    common = lab_b.loc[early].index.intersection(lab_p.loc[early].index)
    np.testing.assert_allclose(lab_b.loc[early].loc[common, "fwd_ret_20d"], lab_p.loc[early].loc[common, "fwd_ret_20d"])
    # labels dated t use t+1 onward, which was perturbed -> they differ
    common = lab_b.loc[t].index.intersection(lab_p.loc[t].index)
    assert not np.allclose(lab_b.loc[t].loc[common, "fwd_ret_5d"], lab_p.loc[t].loc[common, "fwd_ret_5d"])


def test_label_value_matches_manual_computation(runs):
    _, base, _ = runs
    panel, labels, days = base["panel"], base["labels"], base["days"]
    row = labels.dropna(subset=["fwd_ret_5d"]).iloc[len(labels) // 3]
    s = panel[panel["isin"] == row["isin"]].set_index("date")["adj_open_tr"]
    pos = days.get_loc(row["date"])
    expected = s.loc[days[pos + 6]] / s.loc[days[pos + 1]] - 1
    assert row["fwd_ret_5d"] == pytest.approx(expected)
    assert row["label_end_5d"] == days[pos + 6]


def test_labels_unchanged_when_past_prices_change():
    tables, _ = make_fixture(7)
    days = pd.DatetimeIndex(sorted(tables["calendar"]["date"]))
    t = days[400]
    base = _pipeline(tables)
    mod = {k: v.copy() for k, v in tables.items()}
    b = mod["bhavcopy"]
    past = b["date"] <= t
    on_t = b["date"] == t
    b.loc[on_t, "open"] = (b.loc[on_t, "open"] + b.loc[on_t, "close"]) / 2  # open at t is not used by the t label
    b.loc[past & ~on_t, "volume"] *= 1.7  # neither are past volumes
    pert = _pipeline(mod)
    a = base["labels"].set_index(["date", "isin"]).xs(t)["fwd_ret_5d"]
    c = pert["labels"].set_index(["date", "isin"]).xs(t)["fwd_ret_5d"]
    np.testing.assert_allclose(a.to_numpy(), c.reindex(a.index).to_numpy(), rtol=1e-12, equal_nan=True)


def test_end_of_data_labels_are_nan(runs):
    _, base, _ = runs
    lab, days = base["labels"], base["days"]
    tail = lab[lab["date"] >= days[-6]]
    assert tail["fwd_ret_5d"].isna().all()
    assert lab[lab["date"] >= days[-21]]["fwd_ret_20d"].isna().all()
