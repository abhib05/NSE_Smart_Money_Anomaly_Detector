"""DuckDB catalogue over the processed parquet datasets (`data/processed/smd.duckdb`).

Every `<layer>/<name>.parquet` becomes a view `<layer>_<name>` so the whole pipeline can
be queried with SQL, e.g. `SELECT * FROM silver_daily_panel WHERE symbol = 'INFY'`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from smd.config import Config


def refresh_views(cfg: Config) -> Path:
    db_path = cfg.paths.processed / "smd.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    roots = {p.parent.name: p.parent for p in cfg.paths.processed.rglob("*.parquet")}
    if cfg.paths.bronze_override is not None:
        roots["bronze"] = cfg.paths.bronze_override
    con = duckdb.connect(str(db_path))
    try:
        for layer, root in roots.items():
            for pq in sorted(root.glob("*.parquet")):
                view = f"{layer}_{pq.stem}"
                con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{pq.as_posix()}')")
    finally:
        con.close()
    return db_path


def query(cfg: Config, sql: str):
    con = duckdb.connect(str(cfg.paths.processed / "smd.duckdb"), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()
