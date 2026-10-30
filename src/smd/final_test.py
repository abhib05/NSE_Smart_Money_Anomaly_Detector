"""Held-out test evaluation - run once, with frozen choices.

`run_final_test()` evaluates the already-fixed pipeline (same variants, same weighting
schemes, same hyper-parameters, walk-forward models already trained) on dates
>= `test_start`: signal research, ranker IC and backtests. Every evaluation is appended to
`runs/TEST_LOG.jsonl` (config fingerprint, git commit, time) and the report states how
many times the test period was evaluated. A second evaluation of the same configuration
is refused unless `force=True`, which is logged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from smd.config import Config
from smd.utils import get_logger, git_commit, read_dataset

log = get_logger(__name__)


def final_test_log_path(cfg: Config):
    return cfg.paths.runs_dir / "TEST_LOG.jsonl"


def previous_evaluations(cfg: Config) -> list[dict]:
    p = final_test_log_path(cfg)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_final_test(cfg: Config, run, force: bool = False) -> dict:
    from smd.backtest.run import run_backtests
    from smd.clean.panel import trading_days
    from smd.model.ranker import ablation_table, load_model_frame
    from smd.research.run import run_research

    prev = [e for e in previous_evaluations(cfg) if e["config_fingerprint"] == cfg.fingerprint()]
    if prev and not force:
        raise RuntimeError(
            f"the test period was already evaluated {len(prev)} time(s) with this configuration "
            f"(last: run {prev[-1]['run_id']}). Re-running requires --force and will be logged."
        )
    entry = {"run_id": run.run_id, "config_fingerprint": cfg.fingerprint(), "git_commit": git_commit(),
             "at": datetime.now(UTC).isoformat(timespec="seconds"), "forced": bool(prev)}
    p = final_test_log_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.warning("evaluating the held-out test period (evaluation #%d for this config)", len(prev) + 1)

    research = run_research(cfg, run, "test")
    days = trading_days(cfg)
    df = load_model_frame(cfg, days)
    preds = read_dataset(cfg, "gold", "predictions")
    out = run.dir / "model"
    out.mkdir(parents=True, exist_ok=True)
    table = ablation_table(cfg, preds, df, "test")
    table.to_csv(out / "ablation_test.csv", index=False)
    bt = run_backtests(cfg, run, "test")
    run.update(test_evaluation=entry)
    return {"research": research, "ablation_test_mean_ic_5d": table.set_index("variant")["mean_ic_5d"].round(4).to_dict(),
            "backtest": bt, "evaluation_number": len(prev) + 1}
