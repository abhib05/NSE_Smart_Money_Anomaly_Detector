"""Walk-forward cross-validation with purging and embargo.

Fold boundaries come from skfolio's `WalkForward` on the trading calendar (expanding
train window, retrain every `retrain_every_months`). Rows are then assigned to folds by
date, and training rows are **purged** (label window `[t+1, label_end]` overlaps the test
window) and **embargoed** (dated within `embargo_days` trading days before the test start).
A fold whose test window straddles `test_start` is split in two so that every fold lies
entirely in the development or the held-out test period.

`cpcv_date_folds()` wraps skfolio's `CombinatorialPurgedCV` for robustness checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def period(self, test_start: pd.Timestamp) -> str:
        return "test" if self.test_start >= test_start else "dev"


def walk_forward_folds(days: pd.DatetimeIndex, start: pd.Timestamp, first_train_months: int,
                       retrain_every_months: int, test_start: pd.Timestamp | None = None) -> list[Fold]:
    d = days[days >= start]
    X = pd.DataFrame(np.zeros((len(d), 1)), index=d)
    cv = WalkForward(
        test_size=1, train_size=pd.DateOffset(months=first_train_months), freq=f"{retrain_every_months}MS",
        expand_train=True, reduce_test=True,
    )
    folds: list[Fold] = []
    for tr, te in cv.split(X):
        bounds = [(d[te[0]], d[te[-1]])]
        if test_start is not None and d[te[0]] < test_start <= d[te[-1]]:
            before = d[(d >= d[te[0]]) & (d < test_start)]
            after = d[(d >= test_start) & (d <= d[te[-1]])]
            bounds = [(before[0], before[-1]), (after[0], after[-1])]
        for ts, tend in bounds:
            folds.append(Fold(len(folds), d[tr[0]], d[d < ts][-1], ts, tend))
    return folds


def train_mask(sample_dates: pd.Series, label_end: pd.Series | None, fold: Fold, days: pd.DatetimeIndex,
               embargo_days: int) -> pd.Series:
    """Rows usable for training in `fold`: dated in the train window, label window ending
    before the test start (purge), and not within `embargo_days` before it (embargo)."""
    pos = days.searchsorted(fold.test_start)
    embargo_cut = days[max(0, pos - embargo_days)]
    m = (sample_dates >= fold.train_start) & (sample_dates < embargo_cut)
    if label_end is not None:
        m &= pd.to_datetime(label_end) < fold.test_start
    return m


def in_test_window(sample_dates: pd.Series, fold: Fold) -> pd.Series:
    return (sample_dates >= fold.test_start) & (sample_dates <= fold.test_end)


def cpcv_date_folds(dates: pd.DatetimeIndex, n_folds: int = 6, n_test_folds: int = 2,
                    purged_size: int = 5, embargo_size: int = 20) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """CombinatorialPurgedCV over a sorted array of unique dates.
    Returns (train_date_positions, [test_date_positions per test block])."""
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds, purged_size=purged_size,
                               embargo_size=embargo_size)
    X = np.zeros((len(dates), 1))
    return [(tr, list(te)) for tr, te in cv.split(X)]
