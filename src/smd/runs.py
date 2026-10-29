"""Run manifests: every pipeline run gets `runs/<run_id>/manifest.json`.

A run id is `YYYYmmdd-HHMMSS-<config fingerprint>`. `runs/LATEST` points at the most
recent run so stage commands without `--run` append to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smd.config import Config
from smd.utils import data_snapshot_hash, git_commit


@dataclass
class Run:
    run_id: str
    dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def update(self, **fields: Any) -> None:
        m = self.load_manifest()
        for k, v in fields.items():
            if isinstance(v, dict) and isinstance(m.get(k), dict):
                m[k].update(v)
            else:
                m[k] = v
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")

    def record_stage(self, cfg: Config, stage: str, metrics: dict[str, Any] | None = None) -> None:
        m = self.load_manifest()
        stages = m.get("stages", [])
        stages.append(
            {
                "stage": stage,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "data_snapshot": data_snapshot_hash(cfg),
            }
        )
        self.update(stages=stages, **({"metrics": {stage: metrics}} if metrics else {}))

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def new_run(cfg: Config, command: str) -> Run:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + cfg.fingerprint()[:6]
    run = Run(run_id, cfg.paths.runs_dir / run_id)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.update(
        run_id=run_id,
        command=command,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        config_files=cfg.config_files,
        config=cfg.model_dump(mode="json"),
        config_fingerprint=cfg.fingerprint(),
        data_snapshot=data_snapshot_hash(cfg),
        seed=cfg.seed,
    )
    (cfg.paths.runs_dir / "LATEST").write_text(run_id, encoding="utf-8")
    return run


def get_run(cfg: Config, run_id: str | None, command: str = "") -> Run:
    """Return run `run_id`, else the LATEST run, else a new run."""
    if run_id:
        d = cfg.paths.runs_dir / run_id
        if not d.exists():
            raise FileNotFoundError(f"run {run_id} not found under {cfg.paths.runs_dir}")
        return Run(run_id, d)
    latest = cfg.paths.runs_dir / "LATEST"
    if latest.exists():
        rid = latest.read_text(encoding="utf-8").strip()
        if (cfg.paths.runs_dir / rid).exists():
            return Run(rid, cfg.paths.runs_dir / rid)
    return new_run(cfg, command)
