"""`smd` command-line interface."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer

from smd.config import Config, load_config
from smd.runs import get_run, new_run
from smd.utils import get_logger, seed_everything, setup_logging

app = typer.Typer(add_completion=False, help="NSE Smart-Money Anomaly Detector pipeline")
log = get_logger("smd")

ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="override YAML merged over configs/*.yaml")]
RunOpt = Annotated[str | None, typer.Option("--run", help="run id (default: runs/LATEST)")]

ALL_INGEST = "bhavcopy,delivery,index,bulk,block,corp_actions,symbolchange"


def _setup(config: str | None) -> Config:
    cfg = load_config(config)
    setup_logging(logfile=cfg.paths.runs_dir / "smd.log")
    seed_everything(cfg.seed)
    return cfg


@app.command()
def ingest(
    datasets: Annotated[str, typer.Option(help="comma-separated datasets")] = ALL_INGEST,
    from_: Annotated[str | None, typer.Option("--from", help="start date YYYY-MM-DD")] = None,
    to: Annotated[str | None, typer.Option(help="end date YYYY-MM-DD")] = None,
    bronze: Annotated[bool, typer.Option(help="rebuild bronze tables after downloading")] = True,
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """Download raw NSE files (idempotent, throttled) and build bronze tables."""
    from smd.ingest.bronze import build_bronze
    from smd.ingest.download import ingest as do_ingest

    cfg = _setup(config)
    r = get_run(cfg, run, "ingest")
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    stats = do_ingest(
        cfg, names,
        start=date.fromisoformat(from_) if from_ else None,
        end=date.fromisoformat(to) if to else None,
    )
    log.info("download stats: %s", stats)
    out: dict = {"download": stats}
    if bronze:
        out["bronze"] = build_bronze(cfg)
    r.record_stage(cfg, "ingest", out)


@app.command("bronze")
def bronze_cmd(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Rebuild bronze tables from the raw files already on disk."""
    from smd.ingest.bronze import build_bronze

    cfg = _setup(config)
    counts = build_bronze(cfg)
    get_run(cfg, run, "bronze").record_stage(cfg, "bronze", counts)


@app.command()
def clean(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Bronze -> silver: ISIN chains, corporate-action adjustment, total-return series, DQ checks."""
    from smd.clean.panel import build_silver

    cfg = _setup(config)
    stats = build_silver(cfg)
    log.info("silver: %s", stats)
    get_run(cfg, run, "clean").record_stage(cfg, "clean", stats)


@app.command()
def spotcheck(
    symbol: Annotated[str, typer.Argument(help="NSE symbol")],
    around: Annotated[str, typer.Argument(help="date YYYY-MM-DD (e.g. an ex-date)")],
    window: int = 3,
    config: ConfigOpt = None,
) -> None:
    """Print raw vs adjusted prices around a date (manual corporate-action check)."""
    import pandas as pd

    from smd.clean.panel import spot_check

    cfg = _setup(config)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        typer.echo(spot_check(cfg, symbol, around, window).to_string(index=False))


@app.command()
def universe(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Point-in-time liquidity universe."""
    from smd.pipeline import run_universe

    cfg = _setup(config)
    stats = run_universe(cfg)
    log.info("universe: %s", stats)
    get_run(cfg, run, "universe").record_stage(cfg, "universe", stats)


@app.command()
def features(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Feature registry -> gold features + anomaly behaviour channels."""
    from smd.pipeline import run_features

    cfg = _setup(config)
    stats = run_features(cfg)
    log.info("features: %d rows", stats["rows"])
    get_run(cfg, run, "features").record_stage(cfg, "features", stats)


@app.command()
def labels(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Forward-return labels (open t+1 -> open t+1+h, total return, excess vs EW universe)."""
    from smd.pipeline import run_labels

    cfg = _setup(config)
    stats = run_labels(cfg)
    log.info("labels: %s", stats)
    get_run(cfg, run, "labels").record_stage(cfg, "labels", stats)


@app.command()
def anomaly(
    model: Annotated[str, typer.Option(help="lstm_ae|iforest|all (all models are trained together)")] = "all",
    fold: Annotated[str, typer.Option(help="'all' walk-forward folds")] = "all",
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """Walk-forward LSTM autoencoder + Isolation Forest anomaly and accumulation scores."""
    from smd.anomaly.scoring import MODELS, run_anomaly

    cfg = _setup(config)
    r = get_run(cfg, run, "anomaly")
    models = MODELS if model == "all" else (model, "delivery_z_rule")
    stats = run_anomaly(cfg, models, run=r)
    log.info("anomaly: %s", stats)
    r.record_stage(cfg, "anomaly", stats)


@app.command()
def research(
    period: Annotated[str, typer.Option(help="dev (default) | test - test is reserved for `smd final-test`")] = "dev",
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """Signal research: rank IC, quintiles, event study, robustness (development period)."""
    from smd.research.run import run_research

    if period != "dev":
        raise typer.BadParameter("only the development period here; the test period is evaluated by `smd final-test`")
    cfg = _setup(config)
    r = get_run(cfg, run, "research")
    stats = run_research(cfg, r, period)
    log.info("research: %s", stats)
    r.record_stage(cfg, "research", stats)


@app.command()
def train(
    variant: Annotated[str, typer.Option(help="base | base+delivery | base+delivery+anomaly | all")] = "all",
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """LightGBM ranker variants with purged walk-forward CV; ablation table on dev folds."""
    from smd.model.ranker import VARIANTS, run_train

    cfg = _setup(config)
    r = get_run(cfg, run, "train")
    variants = VARIANTS if variant == "all" else (variant,)
    stats = run_train(cfg, r, variants)
    log.info("train: %s", stats)
    r.record_stage(cfg, "train", stats)


@app.command()
def backtest(
    variant: Annotated[str, typer.Option(help="ranker variant or 'all'")] = "all",
    weighting: Annotated[str, typer.Option(help="equal | hrp | all")] = "all",
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """Cost-aware weekly backtest on the development period (all variants by default)."""
    from smd.backtest.run import WEIGHTINGS, run_backtests
    from smd.model.ranker import VARIANTS

    cfg = _setup(config)
    r = get_run(cfg, run, "backtest")
    variants = VARIANTS if variant == "all" else (variant,)
    weightings = WEIGHTINGS if weighting == "all" else (weighting,)
    stats = run_backtests(cfg, r, "dev", variants, weightings)
    log.info("backtest: %s", stats)
    r.record_stage(cfg, "backtest", stats)


@app.command("final-test")
def final_test(
    force: Annotated[bool, typer.Option(help="re-evaluate a configuration already evaluated (logged)")] = False,
    config: ConfigOpt = None,
    run: RunOpt = None,
) -> None:
    """Evaluate the frozen pipeline ONCE on the held-out test period (logged in runs/TEST_LOG.jsonl)."""
    from smd.final_test import run_final_test

    cfg = _setup(config)
    r = get_run(cfg, run, "final-test")
    stats = run_final_test(cfg, r, force)
    log.info("final test: %s", stats)
    r.record_stage(cfg, "final_test", stats)


@app.command()
def report(config: ConfigOpt = None, run: RunOpt = None) -> None:
    """Build reports/<run_id>/REPORT.md and REPORT.html from a run's artifacts."""
    from smd.report.build import build_report

    cfg = _setup(config)
    r = get_run(cfg, run, "report")
    path = build_report(cfg, r)
    typer.echo(f"report: {path}")


@app.command("all")
def all_cmd(
    final: Annotated[bool, typer.Option("--final-test/--no-final-test", help="also evaluate the test period")] = True,
    force_final: Annotated[bool, typer.Option(help="pass --force to final-test")] = False,
    in_process: Annotated[bool, typer.Option(help="run stages in this process (small data / tests)")] = False,
    config: ConfigOpt = None,
) -> None:
    """Full pipeline end to end in a new run: ingest -> ... -> backtest -> final test -> report.

    Each stage runs in its own Python process (same run id), so peak memory is one stage's,
    not the sum of all stages' leftovers - the full NSE panel needs this on a 16 GB laptop."""
    import subprocess
    import sys

    cfg = _setup(config)
    r = new_run(cfg, "all")
    log.info("run %s", r.run_id)
    stages: list[list[str]] = []
    if cfg.paths.bronze_override is None:
        stages.append(["ingest"])
    else:
        log.info("using bronze tables from %s (no download)", cfg.paths.bronze_override)
    stages += [["clean"], ["universe"], ["features"], ["labels"], ["anomaly"], ["research"], ["train"], ["backtest"]]
    if final:
        stages.append(["final-test", *(["--force"] if force_final else [])])
    stages += [["db"], ["report"]]
    for stage in stages:
        log.info("=== %s ===", stage[0])
        args = [*stage, *(["--config", config] if config else []), *(["--run", r.run_id] if stage[0] != "db" else [])]
        if in_process:
            app(args, standalone_mode=False)
            continue
        res = subprocess.run([sys.executable, "-m", "smd.cli", *args], check=False)
        if res.returncode != 0:
            log.error("stage %s failed (exit %d); resume with `smd %s --run %s`", stage[0], res.returncode, stage[0], r.run_id)
            raise typer.Exit(code=res.returncode)


@app.command()
def db(config: ConfigOpt = None) -> None:
    """(Re)create DuckDB views over every processed parquet dataset."""
    from smd.db import refresh_views

    cfg = _setup(config)
    typer.echo(f"DuckDB catalogue: {refresh_views(cfg)}")


@app.command("new-run")
def new_run_cmd(config: ConfigOpt = None) -> None:
    """Start a new run (subsequent stage commands append to it)."""
    cfg = _setup(config)
    typer.echo(new_run(cfg, "new-run").run_id)


if __name__ == "__main__":  # pragma: no cover
    app()
