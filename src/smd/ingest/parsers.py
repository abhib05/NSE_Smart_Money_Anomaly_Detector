"""Parsers: raw NSE file bytes -> canonical bronze frames.

Canonical bhavcopy columns (both the pre-2024 CSV and the UDiFF format map onto these):
`date, symbol, series, isin, open, high, low, close, last, prev_close, volume,
traded_value, trades`.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

BHAV_COLUMNS = [
    "date", "symbol", "series", "isin", "open", "high", "low", "close", "last",
    "prev_close", "volume", "traded_value", "trades",
]
DELIVERY_COLUMNS = ["date", "symbol", "series", "traded_qty", "deliverable_qty", "delivery_pct"]
DEAL_COLUMNS = ["date", "symbol", "client_name", "side", "qty", "price", "deal_type"]
INDEX_COLUMNS = ["date", "index_name", "open", "high", "low", "close"]
CA_COLUMNS = ["symbol", "series", "isin", "ex_date", "record_date", "subject", "face_value"]


def _unzip_csv(content: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        nested = [n for n in names if n.lower().endswith(".zip")]
        if nested:
            return _unzip_csv(zf.read(nested[0]))
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError("no CSV inside archive")
        return zf.read(csvs[0])


def _maybe_unzip(content: bytes) -> bytes:
    return _unzip_csv(content) if content[:4] == b"PK\x03\x04" else content


def _num(s: pd.Series) -> pd.Series:
    """Parse numbers that may carry Indian digit grouping ("4,90,749"), spaces or '-'."""
    if s.dtype.kind in "if":
        return s.astype(float)
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def detect_bhavcopy_schema(content: bytes) -> str:
    head = _maybe_unzip(content)[:300].decode("utf-8", "replace").lstrip("﻿")
    if head.startswith("TradDt"):
        return "udiff"
    if head.upper().startswith("SYMBOL,SERIES"):
        return "old"
    raise ValueError(f"unrecognised bhavcopy header: {head[:60]!r}")


def parse_bhavcopy(content: bytes) -> pd.DataFrame:
    schema = detect_bhavcopy_schema(content)
    return parse_bhavcopy_udiff(content) if schema == "udiff" else parse_bhavcopy_old(content)


def parse_bhavcopy_old(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_maybe_unzip(content)), dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("UNNAMED")]]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TIMESTAMP"].str.strip(), format="%d-%b-%Y"),
            "symbol": df["SYMBOL"].str.strip(),
            "series": df["SERIES"].str.strip(),
            "isin": df["ISIN"].str.strip(),
            "open": _num(df["OPEN"]),
            "high": _num(df["HIGH"]),
            "low": _num(df["LOW"]),
            "close": _num(df["CLOSE"]),
            "last": _num(df["LAST"]),
            "prev_close": _num(df["PREVCLOSE"]),
            "volume": _num(df["TOTTRDQTY"]),
            "traded_value": _num(df["TOTTRDVAL"]),
            "trades": _num(df["TOTALTRADES"]) if "TOTALTRADES" in df else np.nan,
        }
    )
    return out[BHAV_COLUMNS]


def parse_bhavcopy_udiff(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_maybe_unzip(content)), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" in df:
        df = df[df["FinInstrmTp"].str.strip() == "STK"]
    if "Sgmt" in df:
        df = df[df["Sgmt"].str.strip() == "CM"]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TradDt"].str.strip(), format="%Y-%m-%d"),
            "symbol": df["TckrSymb"].str.strip(),
            "series": df["SctySrs"].str.strip(),
            "isin": df["ISIN"].str.strip(),
            "open": _num(df["OpnPric"]),
            "high": _num(df["HghPric"]),
            "low": _num(df["LwPric"]),
            "close": _num(df["ClsPric"]),
            "last": _num(df["LastPric"]),
            "prev_close": _num(df["PrvsClsgPric"]),
            "volume": _num(df["TtlTradgVol"]),
            "traded_value": _num(df["TtlTrfVal"]),
            "trades": _num(df["TtlNbOfTxsExctd"]),
        }
    )
    return out[BHAV_COLUMNS].reset_index(drop=True)


def parse_delivery_mto(content: bytes) -> pd.DataFrame:
    """MTO_ddmmyyyy.DAT: 4 header lines, then `20,<sr>,<symbol>,<series>,<traded>,<deliverable>,<pct>`.
    The trade date is read from the `Trade Date <15-MAR-2016>` header line."""
    text = content.decode("utf-8", "replace").replace("\r", "")
    trade_date = None
    rows = []
    for line in text.split("\n"):
        if trade_date is None and "Trade Date <" in line:
            trade_date = pd.to_datetime(line.split("Trade Date <", 1)[1].split(">", 1)[0], format="%d-%b-%Y")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7 and parts[0] == "20":
            rows.append(parts[2:7])
    if trade_date is None:
        raise ValueError("MTO file without a 'Trade Date' header")
    df = pd.DataFrame(rows, columns=["symbol", "series", "traded_qty", "deliverable_qty", "delivery_pct"])
    for c in ("traded_qty", "deliverable_qty", "delivery_pct"):
        df[c] = _num(df[c])
    df.insert(0, "date", trade_date)
    return df[DELIVERY_COLUMNS]


def parse_deals_csv(content: bytes, deal_type: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str, encoding="utf-8-sig")
    df.columns = [" ".join(c.split()).lower() for c in df.columns]
    if df.empty:
        return pd.DataFrame(columns=DEAL_COLUMNS)
    col = {c.split(" ")[0]: c for c in df.columns}
    side_col = next(c for c in df.columns if c.startswith("buy"))
    price_col = next(c for c in df.columns if "price" in c)
    qty_col = next(c for c in df.columns if c.startswith("quantity"))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[col["date"]].str.strip(), format="%d-%b-%Y"),
            "symbol": df[col["symbol"]].str.strip(),
            "client_name": df[col["client"]].str.strip().str.upper(),
            "side": df[side_col].str.strip().str.upper(),
            "qty": _num(df[qty_col]),
            "price": _num(df[price_col]),
        }
    )
    out["deal_type"] = deal_type
    return out[DEAL_COLUMNS]


def parse_corp_actions_json(content: bytes) -> pd.DataFrame:
    data = json.loads(content)
    if not data:
        return pd.DataFrame(columns=CA_COLUMNS)
    df = pd.DataFrame(data)

    def dt(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s.replace({"-": None}), format="%d-%b-%Y", errors="coerce")

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].str.strip(),
            "series": df.get("series", pd.Series("EQ", index=df.index)).astype(str).str.strip(),
            "isin": df.get("isin", pd.Series(None, index=df.index)),
            "ex_date": dt(df["exDate"]),
            "record_date": dt(df["recDate"]) if "recDate" in df else pd.NaT,
            "subject": df["subject"].astype(str).str.strip(),
            "face_value": _num(df["faceVal"]) if "faceVal" in df else np.nan,
        }
    )
    return out[CA_COLUMNS]


def parse_index_snapshot(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Index Date"].str.strip(), format="%d-%m-%Y"),
            "index_name": df["Index Name"].str.strip(),
            "open": _num(df["Open Index Value"]),
            "high": _num(df["High Index Value"]),
            "low": _num(df["Low Index Value"]),
            "close": _num(df["Closing Index Value"]),
        }
    )
    return out[INDEX_COLUMNS]


def parse_symbolchange(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str, encoding="latin-1")
    df.columns = [c.strip().upper() for c in df.columns]
    return pd.DataFrame(
        {
            "old_symbol": df["OLD SYMBOL"].str.strip(),
            "new_symbol": df["NEW SYMBOL"].str.strip(),
            "effective_date": pd.to_datetime(df["EFFECTIVE DATE"].str.strip(), format="%d-%b-%Y", errors="coerce"),
        }
    )

