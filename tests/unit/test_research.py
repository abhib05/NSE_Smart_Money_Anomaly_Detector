from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from smd.research.event_study import event_study, select_events
from smd.research.ic_analysis import block_bootstrap_mean_ci, daily_rank_ic, ic_summary, quintile_returns


def _frame(n_dates=50, n=30, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=n_dates):
        s = rng.normal(size=n)
        y = signal * s + rng.normal(size=n)
        rows += [dict(date=d, isin=f"S{i}", s=s[i], y=y[i]) for i in range(n)]
    return pd.DataFrame(rows)


def test_rank_ic_matches_scipy():
    df = _frame(n_dates=3)
    ic = daily_rank_ic(df, "s", "y")
    for d, g in df.groupby("date"):
        assert ic[d] == pytest.approx(spearmanr(g["s"], g["y"]).statistic)


def test_ic_detects_signal_and_noise():
    strong = ic_summary(daily_rank_ic(_frame(signal=0.5), "s", "y"), 5, 300, 0)
    noise = ic_summary(daily_rank_ic(_frame(signal=0.0), "s", "y"), 5, 300, 0)
    assert strong["mean_ic"] > 0.3 and strong["ci_low"] > 0
    assert noise["ci_low"] < 0 < noise["ci_high"]


def test_block_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 500)
    lo, hi = block_bootstrap_mean_ci(x, 5, 500, 0)
    assert lo < x.mean() < hi and lo > 0.7 and hi < 1.3


def test_quintiles_monotone_with_signal():
    tab, spread = quintile_returns(_frame(signal=1.0), "s", "y")
    m = tab.mean()
    assert list(m.index) == [1, 2, 3, 4, 5]
    assert m.is_monotonic_increasing
    assert spread.mean() > 0


def test_event_study_recovers_planted_drift():
    days = pd.bdate_range("2020-01-01", periods=200)
    secs = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    er = pd.DataFrame(rng.normal(0, 0.01, (200, 30)), index=days, columns=secs)
    events = []
    for k, i in enumerate(range(40, 150, 10)):
        s = secs[k % 5]
        er.loc[days[i + 1]: days[i + 10], s] += 0.004  # +4% over 10 days after the event
        events.append(dict(date=days[i], isin=s, acc=1.0))
    scores = pd.DataFrame([dict(date=d, isin=s, acc=0.0) for d in days for s in secs])
    scores = scores.set_index(["date", "isin"])
    for e in events:
        scores.loc[(e["date"], e["isin"]), "acc"] = 1.0
    scores = scores.reset_index()
    ev = select_events(scores, "acc", 0.01, 20, days)
    assert len(ev) == len(events)
    pool = scores.assign(med_traded_value=rng.uniform(1, 2, len(scores)), vol_21d=rng.uniform(1, 2, len(scores)))
    res = event_study(ev, er, pool, (-5, 15))
    post10 = res["car"].set_index("rel_day").loc[10, "diff_car"] - res["car"].set_index("rel_day").loc[0, "diff_car"]
    assert post10 == pytest.approx(0.04, abs=0.015)


def test_select_events_dedupes_clusters():
    days = pd.bdate_range("2020-01-01", periods=60)
    scores = pd.DataFrame([dict(date=d, isin="A", acc=1.0) for d in days] + [dict(date=d, isin="B", acc=0.5) for d in days])
    ev = select_events(scores, "acc", 0.5, 20, days)
    assert (ev["isin"] == "A").sum() == 3  # days 0, 21, 42
