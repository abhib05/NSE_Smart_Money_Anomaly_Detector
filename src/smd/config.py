"""Typed configuration.

`load_config()` reads `configs/base.yaml`, `configs/costs.yaml` and `configs/models.yaml`,
then deep-merges an optional override file (e.g. `configs/fixture.yaml`) on top.
Environment variables prefixed `SMD_` override `paths` (e.g. `SMD_DATA_DIR`).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class Paths(BaseModel):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    reports_dir: Path = Path("reports")
    bronze_override: Path | None = None

    def resolve(self, root: Path) -> Paths:
        def r(p: Path | None) -> Path | None:
            if p is None:
                return None
            return p if p.is_absolute() else (root / p)

        return Paths(
            data_dir=r(self.data_dir),  # type: ignore[arg-type]
            runs_dir=r(self.runs_dir),  # type: ignore[arg-type]
            reports_dir=r(self.reports_dir),  # type: ignore[arg-type]
            bronze_override=r(self.bronze_override),
        )

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"


class IngestCfg(BaseModel):
    start: date = date(2014, 1, 1)
    end: date | None = None
    rate_limit_per_sec: float = 2.0
    max_retries: int = 5
    timeout_sec: float = 30
    udiff_cutoff: date = date(2024, 7, 8)
    refetch_recent_days: int = 7


class CleanCfg(BaseModel):
    max_abs_return: float = 0.30
    volume_z_threshold: float = 8.0
    volume_z_lookback: int = 60
    split_ratio_tolerance: float = 0.03
    infer_missing_ca_from_prevclose: bool = True


class UniverseCfg(BaseModel):
    size: int = 500
    liquidity_lookback_days: int = 126
    min_history_days: int = 200
    max_missing_frac: float = 0.05


class FeaturesCfg(BaseModel):
    winsor_pct: float = 0.01
    bulk_days_since_cap: int = 60


class LabelsCfg(BaseModel):
    horizons: list[int] = Field(default_factory=lambda: [5, 20])
    execution: str = "next_open"


class CVCfg(BaseModel):
    first_train_months: int = 36
    retrain_every_months: int = 12
    embargo_days: int = 20
    ranker_first_train_months: int = 24


class PortfolioCfg(BaseModel):
    rebalance: str = "weekly"
    top_k: int = 25
    max_weight: float = 0.10
    sell_rank_buffer: float = 1.5
    notional_inr: float = 1_000_000
    max_adv_participation: float = 0.05
    hrp_lookback_days: int = 126
    drift_tolerance: float = 0.25
    min_trade_inr: float = 2000.0
    sector_cap: float | None = None


class BacktestCfg(BaseModel):
    delisting_haircut: float = 0.10
    stale_price_exit_days: int = 5
    rf_annual: float = 0.065


class AnomalyCfg(BaseModel):
    seq_len: int = 30
    event_top_pct: float = 0.02


class ResearchCfg(BaseModel):
    bootstrap_samples: int = 1000
    event_window: tuple[int, int] = (-20, 40)


class SlippageCfg(BaseModel):
    base_bps: float = 5.0
    impact_bps_at_1pct: float = 10.0


class CostsCfg(BaseModel):
    stt_buy: float = 0.001
    stt_sell: float = 0.001
    brokerage_pct: float = 0.0
    brokerage_cap_inr: float = 20.0
    exchange_txn: float = 0.0000297
    sebi_fee: float = 0.000001
    stamp_duty_buy: float = 0.00015
    gst_rate: float = 0.18
    dp_charge_per_sell_inr: float = 15.93
    slippage: SlippageCfg = Field(default_factory=SlippageCfg)


class LstmAECfg(BaseModel):
    hidden: int = 64
    layers: int = 2
    bottleneck: int = 16
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 15
    patience: int = 3
    max_train_sequences: int = 120_000
    val_frac: float = 0.1
    loss: str = "mse"
    score_last_k: int = 5
    num_threads: int = 8


class IForestCfg(BaseModel):
    n_estimators: int = 300
    max_samples: int = 4096
    max_train_rows: int = 200_000


class LightGBMCfg(BaseModel):
    objective: str = "regression"
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    target: str = "fwd_excess_5d"


class ModelsCfg(BaseModel):
    lstm_ae: LstmAECfg = Field(default_factory=LstmAECfg)
    iforest: IForestCfg = Field(default_factory=IForestCfg)
    lightgbm: LightGBMCfg = Field(default_factory=LightGBMCfg)


class Config(BaseModel):
    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    study_start: date = date(2015, 1, 1)
    dev_end: date = date(2024, 12, 31)
    test_start: date = date(2025, 1, 1)
    clean: CleanCfg = Field(default_factory=CleanCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    labels: LabelsCfg = Field(default_factory=LabelsCfg)
    cv: CVCfg = Field(default_factory=CVCfg)
    portfolio: PortfolioCfg = Field(default_factory=PortfolioCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    anomaly: AnomalyCfg = Field(default_factory=AnomalyCfg)
    research: ResearchCfg = Field(default_factory=ResearchCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    config_files: list[str] = Field(default_factory=list)

    def fingerprint(self) -> str:
        """Short stable hash of the effective configuration."""
        payload = self.model_dump(mode="json", exclude={"config_files", "paths"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(override: str | Path | None = None, config_dir: Path = CONFIG_DIR) -> Config:
    files = [config_dir / "base.yaml", config_dir / "costs.yaml", config_dir / "models.yaml"]
    merged: dict[str, Any] = {}
    used: list[str] = []
    for f in files:
        if f.exists():
            merged = deep_merge(merged, _read_yaml(f))
            used.append(str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f))
    if override is not None:
        op = Path(override)
        if not op.is_absolute() and not op.exists():
            op = REPO_ROOT / op
        if op.resolve() not in {f.resolve() for f in files}:
            merged = deep_merge(merged, _read_yaml(op))
            used.append(str(override))
    env_paths = {k[4:].lower(): v for k, v in os.environ.items() if k.startswith("SMD_") and k.endswith("_DIR")}
    if env_paths:
        merged.setdefault("paths", {}).update(env_paths)
    cfg = Config(**merged)
    cfg.config_files = used
    cfg.paths = cfg.paths.resolve(REPO_ROOT)
    return cfg
