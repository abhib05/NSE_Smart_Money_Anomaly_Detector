"""CV leakage tests: no training row's label window may overlap a test window (+ embargo)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.cv.splits import cpcv_date_folds, in_test_window, train_mask, walk_forward_folds

DAYS = pd.bdate_range("2015-01-01", "2020-12-31")


def test_folds_are_sequential_and_non_overlapping():
    folds = walk_forward_folds(DAYS, DAYS[0], 24, 12)
    assert len(folds) == 4
    for a, b in zip(folds[:-1], folds[1:], strict=True):
        assert a.test_end < b.test_start
        assert b.train_end >= a.test_end  # expanding window
    for f in folds:
        assert f.train_end < f.test_start


def test_fold_is_split_at_test_start():
    folds = walk_forward_folds(DAYS, DAYS[0], 24, 12, test_start=pd.Timestamp("2019-07-01"))
    starts = [f.test_start for f in folds]
    assert pd.Timestamp("2019-07-01") in starts
    for f in folds:
        assert not (f.test_start < pd.Timestamp("2019-07-01") <= f.test_end)


def _panel_rows(h=5):
    dates = pd.Series(np.repeat(DAYS, 3))
    pos = DAYS.get_indexer(dates)
    end = pd.Series(DAYS[np.minimum(pos + 1 + h, len(DAYS) - 1)])
    return dates, end


def test_no_label_overlap_and_embargo():
    embargo = 20
    for h in (5, 20):
        dates, end = _panel_rows(h)
        for fold in walk_forward_folds(DAYS, DAYS[0], 24, 12):
            tr = train_mask(dates, end, fold, DAYS, embargo)
            te = in_test_window(dates, fold)
            assert tr.any() and te.any()
            assert (end[tr] < fold.test_start).all(), "label window overlaps the test window"
            gap = DAYS.searchsorted(fold.test_start) - DAYS.get_indexer(dates[tr]).max()
            assert gap >= embargo, "embargo violated"
            assert not (tr & te).any()


def test_cpcv_returns_purged_splits():
    dates = pd.bdate_range("2018-01-01", periods=600)
    splits = cpcv_date_folds(dates, n_folds=6, n_test_folds=2, purged_size=5, embargo_size=10)
    assert len(splits) == 15  # C(6, 2)
    for tr, tests in splits:
        for te in tests:
            assert len(np.intersect1d(tr, te)) == 0
