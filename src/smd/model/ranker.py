"""LightGBM cross-sectional ranker and the ablation that answers the research question.

Variants (fixed in advance - see DECISIONS.md):
* `base`                   price/volatility + liquidity features
* `base+delivery`          + delivery and bulk/block-deal features
* `base+delivery+anomaly`  + LSTM-AE / Isolation-Forest anomaly and accumulation scores

Training rows are weekly rebalance dates where *all* variants have inputs (anomaly scores
are themselves out-of-sample, so the ranker starts after the first anomaly fold). Features
are rank-normalised per date; the target is the cross-sectional rank of `fwd_excess_5d`.
Walk-forward folds retrain every `retrain_every_months` with purge + embargo.
SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`).
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from smd.config import Config
from smd.cv.splits import Fold, cpcv_date_folds, train_mask, walk_forward_folds
from smd.features.registry import feature_groups, normalize_cross_section
from smd.research.ic_analysis import daily_rank_ic, ic_summary
from smd.utils import get_logger, read_dataset, write_dataset

log = get_logger(__name__)

ANOMALY_FEATURES = [
    "lstm_ae__score", "lstm_ae__accumulation_score", "iforest__score", "iforest__accumulation_score",
]
VARIANTS = ("base", "base+delivery", "base+delivery+anomaly")


def variant_features(variant: str) -> list[str]:
    g = feature_groups()
    cols = g["price"] + g["liquidity"]
    if variant in ("base+delivery", "base+delivery+anomaly"):
        cols += g["delivery"] + g["bulk"]
    if variant == "base+delivery+anomaly":
        cols += ANOMALY_FEATURES
    return cols


def rebalance_dates(days: pd.DatetimeIndex, freq: str = "weekly") -> pd.DatetimeIndex:
    """Last trading day of each week (or month)."""
    s = pd.Series(days, index=days)
    key = days.to_period("W-FRI") if freq == "weekly" else days.to_period("M")
    return pd.DatetimeIndex(s.groupby(key).max().to_numpy())


def load_model_frame(cfg: Config, days: pd.DatetimeIndex) -> pd.DataFrame:
    feats = read_dataset(cfg, "gold", "features")
    labels = read_dataset(cfg, "gold", "labels")
    scores = read_dataset(cfg, "gold", "anomaly_scores")
    sc = scores[scores["model"].isin(["lstm_ae", "iforest"])]
    wide = sc.pivot_table(index=["date", "isin"], columns="model", values=["score", "accumulation_score"])
    wide.columns = [f"{m}__{t}" for t, m in wide.columns]
    rb = rebalance_dates(days, cfg.portfolio.rebalance)
    df = feats[feats["date"].isin(rb)].merge(wide.reset_index(), on=["date", "isin"], how="inner")
    df = df.merge(labels, on=["date", "isin"], how="left")
    return df.sort_values(["date", "isin"]).reset_index(drop=True)


def prepare(df: pd.DataFrame, cols: list[str], target: str, winsor: float) -> tuple[pd.DataFrame, pd.Series]:
    X = normalize_cross_section(df, cols, winsor)
    y = df.groupby("date")[target].rank(pct=True) * 2 - 1
    return X, y


def _model(cfg: Config) -> lgb.LGBMRegressor:
    p = cfg.models.lightgbm
    return lgb.LGBMRegressor(
        objective=p.objective, n_estimators=p.n_estimators, learning_rate=p.learning_rate, num_leaves=p.num_leaves,
        min_child_samples=p.min_child_samples, subsample=p.subsample, subsample_freq=p.subsample_freq,
        colsample_bytree=p.colsample_bytree, reg_lambda=p.reg_lambda, random_state=cfg.seed,
        deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1,
    )


def ranker_folds(cfg: Config, df: pd.DataFrame, days: pd.DatetimeIndex) -> list[Fold]:
    start = pd.Timestamp(df["date"].min())
    d = days[days >= start]
    return walk_forward_folds(d, start, cfg.cv.ranker_first_train_months, cfg.cv.retrain_every_months,
                              pd.Timestamp(cfg.test_start))


def train_predict(cfg: Config, df: pd.DataFrame, days: pd.DatetimeIndex, variants=VARIANTS
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    target = cfg.models.lightgbm.target
    h = int(target.split("_")[-1].rstrip("d"))
    label_end = df[f"label_end_{h}d"]
    folds = ranker_folds(cfg, df, days)
    preds, shap_rows, fold_info = [], [], []
    for variant in variants:
        cols = variant_features(variant)
        X, y = prepare(df, cols, target, cfg.features.winsor_pct)
        for fold in folds:
            tr = train_mask(df["date"], label_end, fold, days, cfg.cv.embargo_days) & y.notna()
            te = (df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)
            if tr.sum() < 100 or te.sum() == 0:
                continue
            model = _model(cfg).fit(X[tr], y[tr])
            p = model.predict(X[te])
            preds.append(pd.DataFrame({"date": df.loc[te, "date"], "isin": df.loc[te, "isin"],
                                       "model_variant": variant, "score": p, "fold_id": fold.fold_id}))
            if fold.period(pd.Timestamp(cfg.test_start)) == "dev":
                contrib = np.asarray(model.booster_.predict(X[te], pred_contrib=True))[:, :-1]
                shap_rows.append(pd.DataFrame({"variant": variant, "fold_id": fold.fold_id, "feature": cols,
                                               "mean_abs_shap": np.abs(contrib).mean(axis=0)}))
            fold_info.append({"variant": variant, "fold_id": fold.fold_id, "n_train": int(tr.sum()),
                              "n_test": int(te.sum()), "train_end": str(fold.train_end.date()),
                              "test": [str(fold.test_start.date()), str(fold.test_end.date())]})
    return pd.concat(preds, ignore_index=True), pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame(), fold_info


def cpcv_robustness(cfg: Config, df: pd.DataFrame, variants=VARIANTS) -> pd.DataFrame:
    """Combinatorial purged CV on the development period: distribution of OOS IC per variant."""
    target = cfg.models.lightgbm.target
    dev = df[df["date"] <= pd.Timestamp(cfg.dev_end)].reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(dev["date"].unique()))
    if len(dates) < 30:
        return pd.DataFrame()
    embargo_obs = int(np.ceil(cfg.cv.embargo_days / 5))
    splits = cpcv_date_folds(dates, n_folds=6, n_test_folds=2, purged_size=1, embargo_size=embargo_obs)
    rows = []
    for variant in variants:
        X, y = prepare(dev, variant_features(variant), target, cfg.features.winsor_pct)
        for k, (tr_pos, te_blocks) in enumerate(splits):
            tr = dev["date"].isin(dates[tr_pos]) & y.notna()
            model = _model(cfg).fit(X[tr], y[tr])
            for te_pos in te_blocks:
                te = dev["date"].isin(dates[te_pos])
                tmp = dev.loc[te, ["date", target]].assign(pred=model.predict(X[te]))
                ic = daily_rank_ic(tmp, "pred", target, min_names=5)
                rows.append({"variant": variant, "split": k, "mean_ic": float(ic.mean())})
    r = pd.DataFrame(rows)
    return r.groupby("variant")["mean_ic"].agg(["mean", "std", "count"]).reset_index()


def ablation_table(cfg: Config, preds: pd.DataFrame, df: pd.DataFrame, period: str) -> pd.DataFrame:
    lab = df[["date", "isin", *[f"fwd_excess_{h}d" for h in cfg.labels.horizons]]]
    m = preds.merge(lab, on=["date", "isin"], how="left")
    if period == "dev":
        m = m[m["date"] <= pd.Timestamp(cfg.dev_end)]
    else:
        m = m[m["date"] >= pd.Timestamp(cfg.test_start)]
    rows = []
    for v in VARIANTS:
        g = m[m["model_variant"] == v]
        row: dict = {"variant": v, "n_features": len(variant_features(v)), "period": period}
        for h in cfg.labels.horizons:
            # weekly rebalance dates: block of ceil(h/5) weeks for the bootstrap
            s = ic_summary(daily_rank_ic(g, "score", f"fwd_excess_{h}d", min_names=5),
                           max(1, int(np.ceil(h / 5))), cfg.research.bootstrap_samples, cfg.seed)
            row.update({f"{k}_{h}d": v2 for k, v2 in s.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run_train(cfg: Config, run, variants=VARIANTS) -> dict:
    from smd.clean.panel import trading_days

    days = trading_days(cfg)
    df = load_model_frame(cfg, days)
    preds, shap, info = train_predict(cfg, df, days, variants)
    write_dataset(cfg, "gold", "predictions", preds)
    out = run.dir / "model"
    out.mkdir(parents=True, exist_ok=True)
    table = ablation_table(cfg, preds, df, "dev")
    table.to_csv(out / "ablation_dev.csv", index=False)
    if len(shap):
        shap.groupby(["variant", "feature"])["mean_abs_shap"].mean().reset_index().sort_values(
            ["variant", "mean_abs_shap"], ascending=[True, False]).to_csv(out / "shap_dev.csv", index=False)
    cp = cpcv_robustness(cfg, df, variants)
    cp.to_csv(out / "cpcv_dev.csv", index=False)
    (out / "folds.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"rows": len(preds), "ablation_dev_mean_ic_5d": table.set_index("variant")["mean_ic_5d"].round(4).to_dict()}
