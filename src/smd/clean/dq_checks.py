"""Data-quality anomaly checks - the project's first use of anomaly detection.

Rule checks (row-level invariants) and statistical checks (unexplained jumps, volume
outliers). Each flag is a boolean column `dq_<name>`; `dq_flags` joins the names of the
flags raised on a row and `dq_exclude` marks rows that must not feed features.

Policy (DECISIONS.md D-007): rule violations and unexplained price jumps are excluded;
`volume_spike` is *reported but not excluded* - extreme volume is exactly the kind of
behaviour this project studies, so dropping it would delete the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.clean.corporate_actions import match_typical_factor

RULE_FLAGS = [
    "high_lt_open_close",
    "low_gt_open_close",
    "negative_volume",
    "delivery_pct_out_of_range",
    "deliverable_gt_traded",
]
STAT_FLAGS = ["large_move_no_ca", "possible_missing_corporate_action", "ca_price_mismatch", "volume_spike"]
INFO_FLAGS = ["ca_inferred_from_prevclose", "div_too_large"]
EXCLUDE_FLAGS = [*RULE_FLAGS, "large_move_no_ca", "possible_missing_corporate_action", "ca_price_mismatch"]
EPS = 1e-9


def rule_checks(p: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=p.index)
    out["high_lt_open_close"] = p["high"] < np.fmax(p["open"], p["close"]) - EPS
    out["low_gt_open_close"] = p["low"] > np.fmin(p["open"], p["close"]) + EPS
    out["negative_volume"] = p["volume"] < 0
    dp = p["delivery_pct"]
    out["delivery_pct_out_of_range"] = dp.notna() & ((dp < 0) | (dp > 100 + EPS))
    out["deliverable_gt_traded"] = p["deliverable_qty"] > p["traded_qty"] + EPS
    return out


def robust_volume_z(p: pd.DataFrame, lookback: int) -> pd.Series:
    """Robust z of log(1+volume) vs the security's *previous* `lookback` rows (median / 1.4826*MAD)."""
    lv = np.log1p(p["adj_volume"].clip(lower=0))
    g = lv.groupby(p["isin"], sort=False)
    med = g.transform(lambda s: s.shift(1).rolling(lookback, min_periods=lookback // 2).median())
    absdev = (lv - med).abs()
    mad = absdev.groupby(p["isin"], sort=False).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback // 2).median()
    )
    return (lv - med) / (1.4826 * mad.replace(0, np.nan))


def stat_checks(p: pd.DataFrame, max_abs_return: float, vol_z_threshold: float, vol_lookback: int,
                tol: float) -> pd.DataFrame:
    """`p` sorted by (isin, date) with adj_close, close, split_event."""
    out = pd.DataFrame(index=p.index)
    g = p.groupby("isin", sort=False)
    adj_ret = p["adj_close"] / g["adj_close"].shift(1) - 1
    raw_ratio = p["close"] / g["close"].shift(1)
    has_ca = p["split_event"] != 1.0
    big = adj_ret.abs() > max_abs_return
    cand = (big & ~has_ca).to_numpy()  # only evaluate the (rare) large moves
    hits = np.zeros(len(p), dtype=bool)
    hits[cand] = [match_typical_factor(r, tol) is not None for r in raw_ratio.to_numpy()[cand]]
    looks_like_ca = pd.Series(hits, index=p.index)
    out["possible_missing_corporate_action"] = big & ~has_ca & looks_like_ca
    out["large_move_no_ca"] = big & ~has_ca & ~looks_like_ca
    # a listed event that does not explain the move (wrong ex-date or partially parsed subject)
    out["ca_price_mismatch"] = big & has_ca
    out["volume_spike"] = robust_volume_z(p, vol_lookback) > vol_z_threshold
    return out


def run_dq(p: pd.DataFrame, max_abs_return: float, vol_z_threshold: float, vol_lookback: int, tol: float) -> pd.DataFrame:
    flags = pd.concat(
        [rule_checks(p), stat_checks(p, max_abs_return, vol_z_threshold, vol_lookback, tol)], axis=1
    ).fillna(False).astype(bool)
    for c in INFO_FLAGS:
        flags[c] = p[c].astype(bool) if c in p else False
    names = np.array(flags.columns)
    arr = flags.to_numpy()
    out = p  # in place
    flags_str = np.full(len(arr), "", dtype=object)
    for i in np.flatnonzero(arr.any(axis=1)):  # only flagged rows need a label
        flags_str[i] = ";".join(names[arr[i]])
    out["dq_flags"] = flags_str
    out["dq_exclude"] = flags[EXCLUDE_FLAGS].any(axis=1).to_numpy()
    for c in flags.columns:
        out[f"dq_{c}"] = flags[c].to_numpy()
    return out


def dq_summary(p: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in p.columns if c.startswith("dq_") and c not in ("dq_flags", "dq_exclude")]
    rows = [
        {"flag": c[3:], "rows": int(p[c].sum()), "pct": 100.0 * p[c].mean(),
         "excluded": c[3:] in EXCLUDE_FLAGS}
        for c in cols
    ]
    rows.append({"flag": "ANY EXCLUDED", "rows": int(p["dq_exclude"].sum()),
                 "pct": 100.0 * p["dq_exclude"].mean(), "excluded": True})
    return pd.DataFrame(rows)
