from __future__ import annotations

from pathlib import Path

import pytest

from smd.config import Config, Paths, load_config

FIXTURES = Path(__file__).parent / "fixtures"
RAW_SAMPLES = FIXTURES / "raw_samples"


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Base config whose data/runs/reports directories live in a temp dir."""
    cfg = load_config()
    cfg.paths = Paths(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", reports_dir=tmp_path / "reports")
    return cfg


@pytest.fixture
def sample_bytes():
    def read(name: str) -> bytes:
        return (RAW_SAMPLES / name).read_bytes()

    return read


@pytest.fixture(scope="session")
def fixture_cfg(tmp_path_factory) -> Config:
    """Fixture config with its processed data in a session temp dir."""
    cfg = load_config("configs/fixture.yaml")
    root = tmp_path_factory.mktemp("fixture_run")
    cfg.paths = Paths(data_dir=root / "data", runs_dir=root / "runs", reports_dir=root / "reports",
                      bronze_override=FIXTURES / "bronze")
    return cfg


@pytest.fixture(scope="session")
def fixture_panel(fixture_cfg):
    from smd.clean.panel import build_silver
    from smd.utils import read_dataset

    build_silver(fixture_cfg)
    return read_dataset(fixture_cfg, "silver", "daily_panel")
