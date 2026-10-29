"""Phase 0 data-availability spike.

Probes every dataset the project needs at sample dates across years and prints a
Markdown table (status, bytes, first header line). Results are recorded in
docs/DATA.md. This script is intentionally standalone (no `smd` imports) so it
can be re-run at any time to re-check NSE archive availability.

Usage:  python scripts/phase0_spike.py > spike_results.md
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date

import httpx

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
ARCH = "https://nsearchives.nseindia.com"

SAMPLE_DATES = [
    date(2012, 3, 15),
    date(2014, 3, 14),
    date(2016, 3, 15),
    date(2018, 3, 15),
    date(2020, 3, 16),
    date(2022, 3, 15),
    date(2024, 3, 15),
    date(2024, 9, 16),
    date(2026, 3, 16),
]


def urls_for(d: date) -> dict[str, str]:
    mon = d.strftime("%b").upper()
    return {
        "bhavcopy_old": f"{ARCH}/content/historical/EQUITIES/{d.year}/{mon}/cm{d:%d}{mon}{d.year}bhav.csv.zip",
        "bhavcopy_udiff": f"{ARCH}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
        "delivery_mto": f"{ARCH}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT",
        "full_bhavcopy": f"{ARCH}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv",
        "index_snapshot": f"https://www.niftyindices.com/Daily_Snapshot/ind_close_all_{d:%d%m%Y}.csv",
    }


def first_line(content: bytes) -> str:
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            content = zf.read(zf.namelist()[0])
    return content.decode("utf-8", "replace").splitlines()[0][:70]


def probe_files(client: httpx.Client) -> list[tuple]:
    rows = []
    for d in SAMPLE_DATES:
        for name, url in urls_for(d).items():
            try:
                r = client.get(url)
                ok = r.status_code == 200
                rows.append((name, d, r.status_code, len(r.content), first_line(r.content) if ok else ""))
            except Exception as exc:  # noqa: BLE001 - spike: record any failure
                rows.append((name, d, "ERR", 0, str(exc)[:60]))
            time.sleep(0.6)
    return rows


def probe_api(client: httpx.Client) -> list[tuple]:
    client.get("https://www.nseindia.com/")  # prime cookies
    rows = []
    api = "https://www.nseindia.com/api"
    for d in SAMPLE_DATES:
        ds = d.strftime("%d-%m-%Y")
        for opt in ("bulk_deals", "block_deals"):
            r = client.get(f"{api}/historicalOR/bulk-block-short-deals", params={"optionType": opt, "from": ds, "to": ds})
            n = len(r.json().get("data", [])) if r.status_code == 200 else -1
            rows.append((opt, d, r.status_code, n, "rows (API caps at 70 per call)"))
            time.sleep(0.8)
    for yr in (2012, 2014, 2016, 2020, 2024):
        r = client.get(
            f"{api}/corporates-corporateActions",
            params={"index": "equities", "from_date": f"01-01-{yr}", "to_date": f"31-12-{yr}"},
        )
        n = len(r.json()) if r.status_code == 200 else -1
        rows.append(("corporate_actions", f"{yr}", r.status_code, n, "rows for calendar year"))
        time.sleep(0.8)
    return rows


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        print("| dataset | date | status | bytes/rows | first line |")
        print("|---|---|---|---|---|")
        for row in probe_files(client) + probe_api(client):
            print("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    print()
    print(json.dumps({"probed_at": date.today().isoformat()}))


if __name__ == "__main__":
    main()
