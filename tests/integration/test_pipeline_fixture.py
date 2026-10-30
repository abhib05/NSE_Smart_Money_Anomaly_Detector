"""End-to-end: `smd all` on the committed synthetic fixture (no downloads), then the
held-out-test guard. This is what CI runs."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from smd.cli import app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("e2e")
    for k in ("DATA", "RUNS", "REPORTS"):
        monkeypatch_module.setenv(f"SMD_{k}_DIR", str(root / k.lower()))
    return root


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_smd_all_on_fixture(env):
    res = CliRunner().invoke(app, ["all", "--in-process", "--config", "configs/fixture.yaml"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    run_id = (env / "runs" / "LATEST").read_text().strip()
    run = env / "runs" / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    stages = [s["stage"] for s in manifest["stages"]]
    for s in ("clean", "universe", "features", "labels", "anomaly", "research", "train", "backtest", "final_test"):
        assert s in stages
    for f in ("research_dev/ic_table.csv", "model/ablation_dev.csv", "model/ablation_test.csv",
              "backtest_dev/metrics.csv", "backtest_test/metrics.csv"):
        assert (run / f).exists(), f
    metrics = pd.read_csv(run / "backtest_dev" / "metrics.csv")
    assert {"Nifty 500 (price index)", "EW universe (TR, no costs)"} <= set(metrics["strategy"])
    assert metrics["deflated_sharpe"].notna().sum() == 6
    report = (env / "reports" / run_id / "REPORT.md").read_text(encoding="utf-8")
    assert "Ranking model ablation" in report and "pending" not in report


def test_final_test_cannot_be_repeated_silently(env):
    res = CliRunner().invoke(app, ["final-test", "--config", "configs/fixture.yaml"])
    assert res.exit_code != 0
    assert "already evaluated" in str(res.exception)
    log = (env / "runs" / "TEST_LOG.jsonl").read_text().splitlines()
    assert len(log) == 1
