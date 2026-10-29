"""pandera schemas for every pipeline stage. `validate(df, name)` raises on violation.

Schemas check structure and hard invariants. *Soft* data-quality problems (e.g.
high < close in a raw row) are not schema errors - they are detected and flagged by
`smd.clean.dq_checks` so they can be counted and reported.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from smd.utils import get_logger

log = get_logger(__name__)

_nonneg = Check.ge(0)
_dt = Column("datetime64[ns]", nullable=False)

BRONZE_BHAVCOPY = DataFrameSchema(
    {
        "date": _dt,
        "symbol": Column(str, nullable=False),
        "series": Column(str, nullable=False),
        "isin": Column(str, nullable=True),
        "open": Column(float, nullable=True),
        "high": Column(float, nullable=True),
        "low": Column(float, nullable=True),
        "close": Column(float, _nonneg, nullable=False),
        "last": Column(float, nullable=True),
        "prev_close": Column(float, nullable=True),
        "volume": Column(float, _nonneg, nullable=True),
        "traded_value": Column(float, _nonneg, nullable=True),
        "trades": Column(float, nullable=True),
        "source_file": Column(str),
    },
    unique=["date", "symbol", "series"],
    strict=True,
    coerce=True,
)

BRONZE_DELIVERY = DataFrameSchema(
    {
        "date": _dt,
        "symbol": Column(str),
        "series": Column(str),
        "traded_qty": Column(float, nullable=True),
        "deliverable_qty": Column(float, nullable=True),
        "delivery_pct": Column(float, nullable=True),
        "source_file": Column(str),
    },
    unique=["date", "symbol", "series"],
    strict=True,
    coerce=True,
)

BRONZE_DEALS = DataFrameSchema(
    {
        "date": _dt,
        "symbol": Column(str),
        "client_name": Column(str, nullable=True),
        "side": Column(str, Check.isin(["BUY", "SELL"])),
        "qty": Column(float, _nonneg),
        "price": Column(float, _nonneg),
        "deal_type": Column(str, Check.isin(["bulk", "block"])),
        "source_file": Column(str),
    },
    strict=True,
    coerce=True,
)

BRONZE_INDEX = DataFrameSchema(
    {
        "date": _dt,
        "index_name": Column(str),
        "open": Column(float, nullable=True),
        "high": Column(float, nullable=True),
        "low": Column(float, nullable=True),
        "close": Column(float, Check.gt(0)),
        "source_file": Column(str),
    },
    unique=["date", "index_name"],
    strict=True,
    coerce=True,
)

BRONZE_CORP_ACTIONS = DataFrameSchema(
    {
        "symbol": Column(str),
        "series": Column(str, nullable=True),
        "isin": Column(str, nullable=True),
        "ex_date": Column("datetime64[ns]", nullable=True),
        "record_date": Column("datetime64[ns]", nullable=True),
        "subject": Column(str),
        "face_value": Column(float, nullable=True),
        "source_file": Column(str),
    },
    strict=True,
    coerce=True,
)

BRONZE_CALENDAR = DataFrameSchema(
    {
        "date": _dt,
        "has_bhavcopy": Column(bool),
        "has_index": Column(bool),
        "has_delivery": Column(bool),
        "is_trading_day": Column(bool),
        "is_gap": Column(bool),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

SILVER_DAILY_PANEL = DataFrameSchema(
    {
        "date": _dt,
        "isin": Column(str, nullable=False),
        "raw_isin": Column(str, nullable=False),
        "symbol": Column(str),
        "close": Column(float, Check.gt(0)),
        "adj_factor": Column(float, Check.gt(0)),
        "tr_factor": Column(float, Check.gt(0)),
        "adj_close": Column(float, Check.gt(0)),
        "adj_close_tr": Column(float, Check.gt(0)),
        "traded_value": Column(float, _nonneg, nullable=True),
        "delivery_pct": Column(float, nullable=True),
        "dq_flags": Column(str),
        "dq_exclude": Column(bool),
    },
    unique=["date", "isin"],
    strict=False,
    coerce=True,
)

SILVER_DEALS = DataFrameSchema(
    {
        "date": _dt,
        "isin": Column(str),
        "symbol": Column(str),
        "client_name": Column(str, nullable=True),
        "side": Column(str, Check.isin(["BUY", "SELL"])),
        "qty": Column(float, _nonneg),
        "price": Column(float, _nonneg),
        "value": Column(float, _nonneg),
        "deal_type": Column(str, Check.isin(["bulk", "block"])),
    },
    strict=True,
    coerce=True,
)

GOLD_UNIVERSE = DataFrameSchema(
    {
        "date": _dt,
        "isin": Column(str),
        "liquidity_rank": Column(int, Check.ge(1)),
        "med_traded_value": Column(float, Check.gt(0)),
        "in_universe": Column(bool),
    },
    unique=["date", "isin"],
    strict=True,
    coerce=True,
)

GOLD_FEATURES = DataFrameSchema(
    {
        "date": _dt,
        "isin": Column(str),
        "delivery_pct": Column(float, Check.in_range(0, 100), nullable=True),
        "delivery_z_xs_rank": Column(float, Check.in_range(-1, 1), nullable=True),
        "quiet_accumulation": Column(float, Check.isin([0.0, 1.0]), nullable=True),
        "bulk_buyers_20d": Column(float, Check.ge(0), nullable=True),
        "bulk_imbalance_20d": Column(float, Check.in_range(-1, 1), nullable=True),
    },
    unique=["date", "isin"],
    strict=False,
    coerce=True,
)

GOLD_LABELS = DataFrameSchema(
    {
        "date": _dt,
        "isin": Column(str),
        "fwd_ret_5d": Column(float, Check.gt(-1), nullable=True),
        "fwd_excess_5d": Column(float, nullable=True),
    },
    unique=["date", "isin"],
    strict=False,
    coerce=True,
)

SCHEMAS: dict[str, DataFrameSchema] = {
    "bronze.bhavcopy": BRONZE_BHAVCOPY,
    "bronze.delivery": BRONZE_DELIVERY,
    "bronze.deals": BRONZE_DEALS,
    "bronze.index": BRONZE_INDEX,
    "bronze.corporate_actions": BRONZE_CORP_ACTIONS,
    "bronze.calendar": BRONZE_CALENDAR,
    "silver.daily_panel": SILVER_DAILY_PANEL,
    "silver.deals": SILVER_DEALS,
    "gold.universe": GOLD_UNIVERSE,
    "gold.features": GOLD_FEATURES,
    "gold.labels": GOLD_LABELS,
}


def register(name: str, schema: DataFrameSchema) -> None:
    SCHEMAS[name] = schema


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate `df` against schema `name`; raises `pandera.errors.SchemaErrors` with all failures."""
    schema = SCHEMAS[name]
    try:
        out = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        log.error("schema %s failed:\n%s", name, exc.failure_cases.head(20))
        raise
    log.debug("schema %s ok (%d rows)", name, len(out))
    return out
