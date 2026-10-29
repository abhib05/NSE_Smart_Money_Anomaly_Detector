"""Walk-forward anomaly scoring: train per fold on the training window, score the next
period out of sample, convert raw scores to cross-sectional percentiles, apply the
accumulation (direction) filter.

    accumulation_score = anomaly_percentile * 1[delivery z > 0  AND  net bulk buys (20d) >= 0
                                                AND  5d return > -1 x 5-day volatility]

Models: `lstm_ae`, `iforest`, and the one-line baseline `delivery_z_rule` (percentile of
the delivery-% z-score) used to check whether the models add anything over a simple rule.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from smd.anomaly import iforest as ifo
from smd.anomaly import lstm_ae
from smd.anomaly.sequences import Cube, build_cube, gather, index_pairs, summary_vectors, valid_windows
from smd.config import Config
from smd.cv.splits import Fold, walk_forward_folds
from smd.utils import get_logger, read_dataset, write_dataset

log = get_logger(__name__)

MODELS = ("lstm_ae", "iforest", "delivery_z_rule")
FILTER_COLUMNS = ["date", "isin", "delivery_pct_z_60d", "bulk_net_buy_20d", "ret_5d", "vol_21d"]


def xs_percentile(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def accumulation_mask(f: pd.DataFrame) -> pd.Series:
    vol5 = f["vol_21d"] * np.sqrt(5)
    return (
        (f["delivery_pct_z_60d"] > 0)
        & (f["bulk_net_buy_20d"].fillna(0) >= 0)
        & (f["ret_5d"] > -vol5)
    ).fillna(False)


def anomaly_folds(cfg: Config, days: pd.DatetimeIndex) -> list[Fold]:
    return walk_forward_folds(days, pd.Timestamp(cfg.study_start), cfg.cv.first_train_months,
                              cfg.cv.retrain_every_months, pd.Timestamp(cfg.test_start))


def _sample_train(cube: Cube, universe: pd.DataFrame, fold: Fold, L: int, n_max: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = universe[(universe["date"] >= fold.train_start) & (universe["date"] <= fold.train_end)]
    t, s = index_pairs(cube, u)
    ok = valid_windows(cube, t, s, L)
    t, s = t[ok], s[ok]
    if len(t) > n_max:
        pick = np.sort(rng.choice(len(t), n_max, replace=False))
        t, s = t[pick], s[pick]
    return t, s


def _split_val(t: np.ndarray, val_frac: float) -> np.ndarray:
    """Time-ordered validation: the last `val_frac` of training days."""
    cut = np.quantile(t, 1 - val_frac) if len(t) else 0
    return t >= cut


def score_fold(cube: Cube, universe: pd.DataFrame, fold: Fold, cfg: Config, model_dir) -> tuple[pd.DataFrame, dict]:
    L = cfg.anomaly.seq_len
    rng = np.random.default_rng(cfg.seed + fold.fold_id)
    test = universe[(universe["date"] >= fold.test_start) & (universe["date"] <= fold.test_end)][["date", "isin"]]
    tt, ts = index_pairs(cube, test)
    ok = valid_windows(cube, tt, ts, L)
    test = test[ok].copy()
    tt, ts = tt[ok], ts[ok]
    info: dict = {"fold_id": fold.fold_id, "train": [str(fold.train_start.date()), str(fold.train_end.date())],
                  "test": [str(fold.test_start.date()), str(fold.test_end.date())], "n_test": int(len(test))}

    # --- LSTM autoencoder
    mc = cfg.models.lstm_ae
    t, s = _sample_train(cube, universe, fold, L, mc.max_train_sequences, rng)
    is_val = _split_val(t, mc.val_frac)
    x_tr, x_va = gather(cube, t[~is_val], s[~is_val], L), gather(cube, t[is_val], s[is_val], L)
    res = lstm_ae.train_autoencoder(x_tr, x_va, mc, cfg.seed + fold.fold_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": res.model.state_dict(), "config": mc.model_dump(), "n_channels": cube.n_channels,
                "seq_len": L, "fold": info}, model_dir / f"lstm_ae_fold{fold.fold_id}.pt")
    info.update(n_train=int(len(x_tr)), n_val=int(len(x_va)), best_epoch=res.best_epoch,
                best_val_mse=res.best_val, history=res.history)
    errs = np.concatenate([
        lstm_ae.reconstruction_errors(res.model, gather(cube, tt[i: i + 20000], ts[i: i + 20000], L), mc.score_last_k)
        for i in range(0, len(tt), 20000)
    ]) if len(tt) else np.array([])
    test["lstm_ae"] = errs

    # --- Isolation Forest (same training sample)
    ic = cfg.models.iforest
    pick = rng.permutation(len(t))[: ic.max_train_rows]
    forest = ifo.fit_iforest(summary_vectors(cube, t[pick], s[pick]), ic, cfg.seed + fold.fold_id)
    test["iforest"] = ifo.iforest_scores(forest, summary_vectors(cube, tt, ts)) if len(tt) else []
    log.info("fold %d %s..%s: LSTM-AE best epoch %d val %.4f; %d test rows",
             fold.fold_id, info["test"][0], info["test"][1], res.best_epoch, res.best_val, len(test))
    return test, info


def run_anomaly(cfg: Config, models: tuple[str, ...] = MODELS, run=None) -> dict:
    from smd.clean.panel import trading_days

    days = trading_days(cfg)
    universe = read_dataset(cfg, "gold", "universe")
    universe = universe[universe["in_universe"]][["date", "isin"]]
    beh = read_dataset(cfg, "gold", "behaviour")
    feats = read_dataset(cfg, "gold", "features", columns=FILTER_COLUMNS)
    cube = build_cube(beh, days)
    folds = anomaly_folds(cfg, days)
    model_dir = cfg.paths.processed / "models"
    frames, infos = [], []
    for fold in folds:
        df, info = score_fold(cube, universe, fold, cfg, model_dir)
        df["fold_id"] = fold.fold_id
        frames.append(df)
        infos.append(info)
    raw = pd.concat(frames, ignore_index=True).merge(feats, on=["date", "isin"], how="left")
    raw["delivery_z_rule"] = raw["delivery_pct_z_60d"]
    acc = accumulation_mask(raw)
    out = []
    for m in models:
        pct = xs_percentile(raw.dropna(subset=[m]), m).reindex(raw.index)
        out.append(pd.DataFrame({
            "date": raw["date"], "isin": raw["isin"], "model": m, "raw_score": raw[m], "score": pct,
            "accumulation_score": pct * acc.astype(float), "fold_id": raw["fold_id"],
        }).dropna(subset=["score"]))
    scores = pd.concat(out, ignore_index=True).sort_values(["model", "date", "isin"]).reset_index(drop=True)
    write_dataset(cfg, "gold", "anomaly_scores", scores)
    if run is not None:
        run.path("anomaly", "folds.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
    return {
        "folds": len(folds),
        "first_oos_date": str(scores["date"].min().date()),
        "rows_per_model": scores.groupby("model").size().to_dict(),
        "best_val_mse": [round(i["best_val_mse"], 4) for i in infos],
    }
