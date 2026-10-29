"""Behaviour-sequence tensors for the anomaly models.

The behaviour channels (`bh_*`, each already standardised with past data only) are laid
out on a calendar-aligned cube `X[channel, day, security]`. The sequence for (t, s) is
`X[:, t-L+1 : t+1, s]` - strictly data up to and including *t*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smd.features.registry import BEHAVIOUR_CHANNELS

CLIP = 8.0


@dataclass
class Cube:
    X: np.ndarray            # (C, T, S) float32, NaN = missing
    days: pd.DatetimeIndex
    secs: pd.Index

    @property
    def n_channels(self) -> int:
        return self.X.shape[0]


def build_cube(behaviour: pd.DataFrame, days: pd.DatetimeIndex) -> Cube:
    secs = pd.Index(sorted(behaviour["isin"].unique()))
    C, T, S = len(BEHAVIOUR_CHANNELS), len(days), len(secs)
    X = np.full((C, T, S), np.nan, dtype=np.float32)
    ti = days.get_indexer(behaviour["date"])
    si = secs.get_indexer(behaviour["isin"])
    ok = ti >= 0
    for c, ch in enumerate(BEHAVIOUR_CHANNELS):
        X[c, ti[ok], si[ok]] = behaviour[ch].to_numpy(np.float32)[ok]
    np.clip(X, -CLIP, CLIP, out=X)
    return Cube(X, days, secs)


def index_pairs(cube: Cube, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map (date, isin) rows to cube positions (-1 when unknown)."""
    return cube.days.get_indexer(pairs["date"]), cube.secs.get_indexer(pairs["isin"])


def valid_windows(cube: Cube, t: np.ndarray, s: np.ndarray, L: int, min_obs: float = 0.8) -> np.ndarray:
    """A window is valid if it fits in the calendar, the last day is observed, and >= min_obs of
    its (day, channel) cells are observed."""
    ok = (t >= L - 1) & (s >= 0) & (t >= 0)
    out = np.zeros(len(t), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    obs_day = ~np.isnan(cube.X).all(axis=0)  # (T, S) any channel observed
    cum = np.vstack([np.zeros((1, obs_day.shape[1])), np.cumsum(obs_day, axis=0)])
    tt, ss = t[idx], s[idx]
    n_obs = cum[tt + 1, ss] - cum[tt + 1 - L, ss]
    last_obs = obs_day[tt, ss]
    out[idx] = last_obs & (n_obs >= min_obs * L)
    return out


def gather(cube: Cube, t: np.ndarray, s: np.ndarray, L: int) -> np.ndarray:
    """Return windows as (N, L, C) float32 with NaN filled by 0 (the standardised mean)."""
    offs = np.arange(-L + 1, 1)
    tt = t[:, None] + offs[None, :]                       # (N, L)
    w = cube.X[:, tt, s[:, None]]                         # (C, N, L)
    w = np.transpose(w, (1, 2, 0))                        # (N, L, C)
    return np.nan_to_num(w, nan=0.0).astype(np.float32, copy=False)


def summary_vectors(cube: Cube, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Isolation-Forest inputs: latest day + 5-day and 20-day means of each channel."""
    last = gather(cube, t, s, 1)[:, -1, :]
    w20 = gather(cube, t, s, 20)
    return np.concatenate([last, w20[:, -5:, :].mean(axis=1), w20.mean(axis=1)], axis=1)
