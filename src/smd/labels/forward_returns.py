"""Forward-return labels.

Timing: features are known at the close of *t*; we trade at the open of *t+1*
(`execution: next_open`, or close of *t+1* with `next_close`).

* `fwd_ret_h`    = P_tr(t+1+h) / P_tr(t+1) - 1 on the total-return-adjusted series, where
  P is the open (or close) and t+k is k *trading days* ahead on the exchange calendar.
  If the stock stops trading inside the window the last available price is used
  (delisting is handled with a haircut in the backtest, not here).
* `fwd_excess_h` = fwd_ret_h minus the equal-weight mean of fwd_ret_h over the universe at t.
* `label_end_h`  = calendar date t+1+h (used for purging in cross-validation).
Windows that run past the end of the data are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(panel: pd.DataFrame, universe: pd.DataFrame, days: pd.DatetimeIndex,
                 horizons: list[int], execution: str = "next_open") -> pd.DataFrame:
    col = {"next_open": "adj_open_tr", "next_close": "adj_close_tr"}[execution]
    p = panel.loc[~panel["dq_exclude"], ["date", "isin", col]]
    px = p.pivot(index="date", columns="isin", values=col).reindex(days)
    uni = universe.loc[universe["in_universe"], ["date", "isin"]]
    px = px.reindex(columns=sorted(uni["isin"].unique()))
    entry = px.shift(-1)
    mask = pd.DataFrame(False, index=days, columns=px.columns)
    arr = mask.to_numpy()
    arr[days.get_indexer(uni["date"]), px.columns.get_indexer(uni["isin"])] = True
    mask = pd.DataFrame(arr, index=days, columns=px.columns)

    out = uni.set_index(["date", "isin"]).copy()
    for h in horizons:
        exit_px = px.ffill(limit=h).shift(-(1 + h))
        fwd = (exit_px / entry - 1.0).where(mask)
        fwd.iloc[len(days) - (1 + h):] = np.nan  # incomplete windows at the end of the data
        ew = fwd.mean(axis=1)
        out[f"fwd_ret_{h}d"] = fwd.stack(future_stack=True).reindex(out.index)
        out[f"fwd_excess_{h}d"] = fwd.sub(ew, axis=0).stack(future_stack=True).reindex(out.index)
        end_pos = np.minimum(np.arange(len(days)) + 1 + h, len(days) - 1)
        end = pd.Series(days[end_pos], index=days)
        out[f"label_end_{h}d"] = end.reindex(out.index.get_level_values("date")).to_numpy()
    return out.reset_index().sort_values(["date", "isin"]).reset_index(drop=True)
