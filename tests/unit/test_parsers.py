"""Parser tests on truncated *real* NSE files (old and UDiFF bhavcopy, MTO, deals, CA, index)."""

from __future__ import annotations

import pandas as pd
import pytest

from smd.ingest import parsers


def test_old_bhavcopy(sample_bytes):
    content = sample_bytes("cm15MAR2016bhav.csv.zip")
    assert parsers.detect_bhavcopy_schema(content) == "old"
    df = parsers.parse_bhavcopy(content)
    assert list(df.columns) == parsers.BHAV_COLUMNS
    assert (df["date"] == pd.Timestamp("2016-03-15")).all()
    row = df.set_index("symbol").loc["20MICRONS"]
    assert row["isin"] == "INE144J01027"
    assert row["close"] == pytest.approx(27.6)
    assert row["volume"] == 17151
    assert row["traded_value"] == pytest.approx(471883.35)


def test_udiff_bhavcopy(sample_bytes):
    content = sample_bytes("BhavCopy_NSE_CM_0_0_0_20240916_F_0000.csv.zip")
    assert parsers.detect_bhavcopy_schema(content) == "udiff"
    df = parsers.parse_bhavcopy(content)
    assert list(df.columns) == parsers.BHAV_COLUMNS
    assert (df["date"] == pd.Timestamp("2024-09-16")).all()
    row = df.set_index("symbol").loc["COROMANDEL"]
    assert row["series"] == "EQ"
    assert row["isin"] == "INE169A01031"
    assert row["open"] == pytest.approx(1694.40)
    assert row["close"] == pytest.approx(1705.10)
    assert row["prev_close"] == pytest.approx(1694.40)
    assert row["volume"] == 225407
    assert row["traded_value"] == pytest.approx(379992698.20)


def test_old_and_new_schemas_share_dtypes(sample_bytes):
    a = parsers.parse_bhavcopy(sample_bytes("cm15MAR2016bhav.csv.zip"))
    b = parsers.parse_bhavcopy(sample_bytes("BhavCopy_NSE_CM_0_0_0_20240916_F_0000.csv.zip"))
    assert dict(a.dtypes) == dict(b.dtypes)


def test_unknown_bhavcopy_header_raises():
    with pytest.raises(ValueError):
        parsers.detect_bhavcopy_schema(b"garbage,header\n1,2")


def test_delivery_mto(sample_bytes):
    df = parsers.parse_delivery_mto(sample_bytes("MTO_15032016.DAT"))
    assert list(df.columns) == parsers.DELIVERY_COLUMNS
    row = df.set_index("symbol").loc["20MICRONS"]
    assert row["series"] == "EQ"
    assert row["traded_qty"] == 17151
    assert row["deliverable_qty"] == 12287
    assert row["delivery_pct"] == pytest.approx(71.64)
    assert (df["date"] == pd.Timestamp("2016-03-15")).all()


def test_delivery_requires_trade_date():
    with pytest.raises(ValueError):
        parsers.parse_delivery_mto(b"20,1,ABC,EQ,10,5,50.0\n")


def test_bulk_deals_csv_indian_number_format(sample_bytes):
    df = parsers.parse_deals_csv(sample_bytes("bulk_20240916.csv"), "bulk")
    assert list(df.columns) == parsers.DEAL_COLUMNS
    first = df.iloc[0]
    assert first["symbol"] == "AARTISURF"
    assert first["side"] == "SELL"
    assert first["qty"] == 62758  # "62,758"
    assert first["price"] == pytest.approx(763.92)
    assert set(df["side"]) <= {"BUY", "SELL"}
    assert (df["deal_type"] == "bulk").all()


def test_empty_deals_file():
    df = parsers.parse_deals_csv(b'"Date ","Symbol ","Security Name ","Client Name ","Buy / Sell ","Quantity Traded ","Trade Price / Wght. Avg. Price ","Remarks "\n', "block")
    assert df.empty and list(df.columns) == parsers.DEAL_COLUMNS


def test_corp_actions_json(sample_bytes):
    df = parsers.parse_corp_actions_json(sample_bytes("ca_2016Q1.json"))
    assert list(df.columns) == parsers.CA_COLUMNS
    greenply = df[df["symbol"] == "GREENPLY"].iloc[0]
    assert greenply["ex_date"] == pd.Timestamp("2016-01-06")
    assert "Split" in greenply["subject"]


def test_index_snapshot(sample_bytes):
    df = parsers.parse_index_snapshot(sample_bytes("ind_close_all_15032016.csv"))
    assert list(df.columns) == parsers.INDEX_COLUMNS
    n50 = df.set_index("index_name").loc["Nifty 50"]
    assert n50["close"] == pytest.approx(7460.6)
