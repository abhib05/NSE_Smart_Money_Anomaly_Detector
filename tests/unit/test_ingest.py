"""Downloader tests with a mocked NSE: idempotency, 404 handling, immutability, bronze build."""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from smd.ingest import sources
from smd.ingest.bronze import build_bronze
from smd.ingest.download import Manifest, ingest
from smd.ingest.http import ThrottledClient
from tests.conftest import RAW_SAMPLES

OLD_BHAV = (RAW_SAMPLES / "cm15MAR2016bhav.csv.zip").read_bytes()
MTO = (RAW_SAMPLES / "MTO_15032016.DAT").read_bytes()
IDX = (RAW_SAMPLES / "ind_close_all_15032016.csv").read_bytes()
DEALS = (RAW_SAMPLES / "bulk_20240916.csv").read_bytes()
CA = (RAW_SAMPLES / "ca_2016Q1.json").read_bytes()

TRADING_DAY = date(2016, 3, 15)


class FakeNSE:
    """Serves the sample files for 2016-03-15 only; every other daily file is a 404."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        if url.rstrip("/") == "https://www.nseindia.com":
            return httpx.Response(200, text="home")
        if "cm15MAR2016bhav" in url:
            return httpx.Response(200, content=OLD_BHAV)
        if "MTO_15032016" in url:
            return httpx.Response(200, content=MTO)
        if "ind_close_all_15032016" in url:
            return httpx.Response(200, content=IDX)
        if "bulk-block-short-deals" in url:
            return httpx.Response(200, content=DEALS)
        if "corporates-corporateActions" in url:
            return httpx.Response(200, content=CA)
        if "symbolchange" in url:
            return httpx.Response(200, text="NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\nX,A,B,01-JAN-2016\n")
        return httpx.Response(404, text="<html>not found</html>")


def _client(fake: FakeNSE) -> ThrottledClient:
    return ThrottledClient(rate_limit_per_sec=0, max_retries=1, transport=httpx.MockTransport(fake), sleep=lambda s: None)


def _run(cfg, fake, today=date(2016, 4, 30)):
    return ingest(cfg, ["bhavcopy", "index", "delivery", "bulk", "corp_actions"],
                  start=date(2016, 3, 14), end=date(2016, 3, 16), client=_client(fake), today=today)


def test_ingest_is_idempotent(tmp_cfg):
    fake = FakeNSE()
    stats1 = _run(tmp_cfg, fake)
    assert stats1["bhavcopy"]["ok"] == 1 and stats1["bhavcopy"]["missing"] == 2
    assert stats1["delivery"]["ok"] == 1  # only requested for the trading day
    n_calls = len(fake.calls)
    stats2 = _run(tmp_cfg, fake)
    assert len(fake.calls) == n_calls, "re-running ingestion must not hit the network"
    assert all(s["requested"] == 0 for s in stats2.values())


def test_recent_404_is_retried_but_old_404_is_not(tmp_cfg):
    fake = FakeNSE()
    _run(tmp_cfg, fake, today=date(2016, 3, 17))  # 404s are only 1-3 days old
    n = len(fake.calls)
    _run(tmp_cfg, fake, today=date(2016, 3, 18))
    assert len(fake.calls) > n


def test_raw_files_and_manifest_checksums(tmp_cfg):
    _run(tmp_cfg, FakeNSE())
    m = Manifest(tmp_cfg.paths.raw).df
    ok = m[m["status"] == "ok"]
    assert len(ok) >= 4
    for _, row in ok.iterrows():
        assert (tmp_cfg.paths.raw / row["relpath"]).exists()
        assert len(row["sha256"]) == 64


def test_html_error_page_is_not_saved_as_data(tmp_cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Resource not found</html>")

    client = ThrottledClient(0, 1, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    stats = ingest(tmp_cfg, ["bhavcopy"], start=TRADING_DAY, end=TRADING_DAY, client=client, today=date(2016, 5, 1))
    assert stats["bhavcopy"]["missing"] == 1
    assert not any(tmp_cfg.paths.raw.rglob("*.zip"))


def test_retry_then_success():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503) if attempts["n"] < 3 else httpx.Response(200, content=b"ok")

    client = ThrottledClient(0, 5, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    res = client.get("https://nsearchives.nseindia.com/x")
    assert res.status == 200 and attempts["n"] == 3


def test_bronze_build_and_calendar(tmp_cfg):
    _run(tmp_cfg, FakeNSE())
    counts = build_bronze(tmp_cfg)
    assert counts["bhavcopy"] > 0 and counts["delivery"] > 0 and counts["deals"] > 0
    cal = pd.read_parquet(tmp_cfg.paths.processed / "bronze" / "calendar.parquet")
    assert cal.loc[cal["is_trading_day"], "date"].tolist() == [pd.Timestamp(TRADING_DAY)]
    assert counts["calendar_gaps"] == 0


def test_request_routing_by_schema_cutoff():
    cut = date(2024, 7, 8)
    assert "historical/EQUITIES" in sources.bhavcopy_request(date(2024, 7, 5), cut).url
    assert "BhavCopy_NSE_CM" in sources.bhavcopy_request(date(2024, 7, 8), cut).url


def test_incomplete_window_gets_asof_name():
    today = date(2026, 10, 29)
    complete = sources.deals_request("bulk", date(2026, 9, 1), date(2026, 9, 30), today)
    partial = sources.deals_request("bulk", date(2026, 10, 1), date(2026, 10, 31), today)
    assert "_asof" not in complete.relpath
    assert partial.relpath.endswith("_asof20261029.csv")
    assert partial.params["to"] == "28-10-2026"
