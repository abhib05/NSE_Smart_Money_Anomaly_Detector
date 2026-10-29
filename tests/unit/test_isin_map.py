from __future__ import annotations

import pandas as pd

from smd.clean.isin_map import build_isin_map, map_symbol_date_to_sid

DAYS = pd.bdate_range("2020-01-01", periods=30)


def _bhav(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "isin"])


def test_isin_change_under_same_symbol_is_linked():
    rows = [(d, "ABC", "OLD1") for d in DAYS[:10]] + [(d, "ABC", "NEW2") for d in DAYS[10:20]]
    m = build_isin_map(_bhav(rows), DAYS)
    assert m["sid"].nunique() == 1 and (m["sid"] == "OLD1").all()


def test_symbol_reuse_after_long_gap_is_not_linked():
    rows = [(d, "ABC", "OLD1") for d in DAYS[:5]] + [(d, "ABC", "NEW2") for d in DAYS[20:25]]
    m = build_isin_map(_bhav(rows), DAYS)
    assert m["sid"].nunique() == 2


def test_symbol_change_keeps_isin():
    rows = [(d, "ABC", "I1") for d in DAYS[:10]] + [(d, "ABCNEW", "I1") for d in DAYS[10:20]]
    m = build_isin_map(_bhav(rows), DAYS)
    assert set(m["sid"]) == {"I1"}
    q = pd.DataFrame({"symbol": ["ABC", "ABCNEW", "ABC"], "date": [DAYS[3], DAYS[15], DAYS[29]]})
    sid = map_symbol_date_to_sid(q, m, max_staleness_days=5)
    assert sid.iloc[0] == "I1" and sid.iloc[1] == "I1" and pd.isna(sid.iloc[2])
