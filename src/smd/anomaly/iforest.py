"""Isolation Forest baseline on behaviour summaries (latest day + 5d/20d channel means)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from smd.config import IForestCfg


def fit_iforest(x_train: np.ndarray, cfg: IForestCfg, seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=cfg.n_estimators, max_samples=min(cfg.max_samples, len(x_train)),
        random_state=seed, n_jobs=-1,
    )
    return model.fit(x_train)


def iforest_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.score_samples(x)
