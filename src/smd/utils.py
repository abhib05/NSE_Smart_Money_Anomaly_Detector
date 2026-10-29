"""Logging, hashing, seeding and processed-dataset storage helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smd.config import REPO_ROOT, Config

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "matplotlib", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by third-party libs
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a dataframe (order-sensitive)."""
    h = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Processed datasets: parquet + sidecar metadata (version = content hash + config hash)
# --------------------------------------------------------------------------------------

def dataset_path(cfg: Config, layer: str, name: str) -> Path:
    if layer == "bronze" and cfg.paths.bronze_override is not None:
        return cfg.paths.bronze_override / f"{name}.parquet"
    return cfg.paths.processed / layer / f"{name}.parquet"


def write_dataset(cfg: Config, layer: str, name: str, df: pd.DataFrame, extra: dict[str, Any] | None = None) -> Path:
    path = cfg.paths.processed / layer / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {
        "name": name,
        "layer": layer,
        "rows": len(df),
        "columns": list(map(str, df.columns)),
        "content_hash": frame_hash(df),
        "config_fingerprint": cfg.fingerprint(),
        "git_commit": git_commit(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def read_dataset(cfg: Config, layer: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = dataset_path(cfg, layer, name)
    if not path.exists():
        raise FileNotFoundError(f"{layer}/{name} not found at {path}; run the upstream pipeline stage first")
    return pd.read_parquet(path, columns=columns)


def dataset_exists(cfg: Config, layer: str, name: str) -> bool:
    return dataset_path(cfg, layer, name).exists()


def data_snapshot_hash(cfg: Config) -> str:
    """Hash of all processed-dataset metadata: identifies the data a run used."""
    h = hashlib.sha256()
    roots = [cfg.paths.processed]
    if cfg.paths.bronze_override is not None:
        roots.append(cfg.paths.bronze_override)
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.parquet")):
            meta = p.with_suffix(".meta.json")
            h.update(str(p.relative_to(root)).encode())
            h.update(meta.read_bytes() if meta.exists() else str(p.stat().st_size).encode())
    return h.hexdigest()[:16]
