from __future__ import annotations

from smd.config import deep_merge, load_config
from smd.runs import get_run, new_run


def test_base_config_loads_all_files():
    cfg = load_config()
    assert cfg.universe.size == 500
    assert cfg.costs.stt_buy == 0.001
    assert cfg.anomaly.seq_len == 30
    assert cfg.test_start > cfg.dev_end


def test_override_merges_deeply(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("universe:\n  size: 15\n")
    cfg = load_config(p)
    assert cfg.universe.size == 15
    assert cfg.universe.liquidity_lookback_days == 126


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_fingerprint_changes_with_config(tmp_path):
    a = load_config()
    p = tmp_path / "o.yaml"
    p.write_text("seed: 7\n")
    assert a.fingerprint() != load_config(p).fingerprint()


def test_run_manifest(tmp_cfg):
    r = new_run(tmp_cfg, "test")
    m = r.load_manifest()
    for key in ("run_id", "config", "git_commit", "data_snapshot", "seed"):
        assert key in m
    assert get_run(tmp_cfg, None).run_id == r.run_id
    r.record_stage(tmp_cfg, "stage1", {"x": 1})
    assert r.load_manifest()["metrics"]["stage1"] == {"x": 1}
