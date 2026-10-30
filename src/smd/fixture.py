"""Deterministic synthetic market used by tests and CI (no NSE downloads).

`make_fixture()` returns bronze-format tables for ~24 stocks x 3 years with known ground
truth planted in them:

* STK03: 10 -> 2 face-value split with a **new ISIN** (tests the ISIN chain + adjustment)
* STK07: 1:1 bonus (same ISIN)
* STK11: 5 -> 1 split that is **missing from the corporate-action list** (must be flagged
  `possible_missing_corporate_action` and excluded)
* STK15: symbol change to STK15N (same ISIN)
* STK20: delists two-thirds of the way through; STK24 lists after ~1 year
* STK05: one row with high < close; STK09: one 45% bad print that reverts next day
* accumulation events (delivery spike, volume up, muted price, bulk buys) followed by a
  positive drift; distribution events (delivery spike + bulk sells) followed by a negative drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_STOCKS = 24
START, END = "2021-01-01", "2023-12-29"


@dataclass
class Truth:
    accumulation: list[tuple[str, pd.Timestamp]] = field(default_factory=list)
    distribution: list[tuple[str, pd.Timestamp]] = field(default_factory=list)


def _days(seed: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(START, END)
    rng = np.random.default_rng(seed + 1)
    holidays = rng.choice(len(days), size=30, replace=False)
    return days.delete(np.sort(holidays))


def make_fixture(seed: int = 7) -> tuple[dict[str, pd.DataFrame], Truth]:
    rng = np.random.default_rng(seed)
    days = _days(seed)
    n = len(days)
    truth = Truth()
    mkt = rng.normal(0.0004, 0.009, n)
    symbols = [f"STK{i:02d}" for i in range(1, N_STOCKS + 1)]
    bhav_rows, deliv_rows, deal_rows, ca_rows = [], [], [], []

    for k, sym in enumerate(symbols):
        isin = f"INE{k + 100:03d}A01011"
        beta = rng.uniform(0.6, 1.4)
        idio_sd = rng.uniform(0.012, 0.025)
        start_i = 250 if sym == "STK24" else 0
        end_i = int(n * 0.66) if sym == "STK20" else n
        vol_level = np.exp(rng.uniform(np.log(5e4), np.log(2e6)))
        base_deliv = rng.uniform(30, 60)

        ret = beta * mkt + rng.normal(0, idio_sd, n)
        vol_mult = np.exp(rng.normal(0, 0.35, n))
        deliv_pct = base_deliv + rng.normal(0, 6, n)
        bulk_buy = np.zeros(n)
        bulk_sell = np.zeros(n)

        # planted events (kept away from the edges)
        n_events = 4
        ev_days = rng.choice(np.arange(max(start_i, 80), end_i - 45), size=n_events, replace=False)
        for j, e in enumerate(sorted(ev_days)):
            kind = "acc" if (j + k) % 3 != 0 else "dist"
            sl = slice(e - 2, e + 1)
            deliv_pct[sl] += 30
            vol_mult[sl] *= 2.5
            ret[sl] *= 0.3
            drift = 0.004 if kind == "acc" else -0.004
            ret[e + 1: e + 21] += drift
            (bulk_buy if kind == "acc" else bulk_sell)[e] = 1
            (truth.accumulation if kind == "acc" else truth.distribution).append((sym, days[e]))
        deliv_pct = np.clip(deliv_pct, 3, 100)

        # dividends: a cash amount once a year
        div_days = [i for i in range(start_i + 120, end_i, 250)]
        price = np.empty(n)
        p0 = np.exp(rng.uniform(np.log(50), np.log(2000)))
        econ = p0 * np.exp(np.cumsum(ret))  # total-return path
        div_amt = {}
        cum_div_drop = np.ones(n)
        for d in div_days:
            amt = round(float(econ[d - 1] * 0.02), 2)
            div_amt[d] = amt
            cum_div_drop[d:] *= 1 - amt / econ[d - 1]
        price = econ * cum_div_drop

        # share-count events (raw price = economic price * factor after the event)
        share_events: list[tuple[int, float, str, bool]] = []  # (day, price factor, subject, listed)
        if sym == "STK03":
            share_events.append((300, 0.2, "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", True))
        if sym == "STK07":
            share_events.append((400, 0.5, "Bonus 1:1", True))
        if sym == "STK11":
            share_events.append((500, 0.2, "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share", False))
        raw_mult = np.ones(n)
        for d, f, _, _ in share_events:
            raw_mult[d:] *= f
        raw_close = price * raw_mult

        for i in range(start_i, end_i):
            d = days[i]
            c = raw_close[i]
            if i == start_i:
                prev = c / (1 + ret[i])
            else:
                prev = raw_close[i - 1]  # NSE reports the *raw* previous close, also on ex-dates (D-018)
            o = prev * np.exp(rng.normal(0, 0.004)) if i > start_i else c
            h = max(o, c) * (1 + abs(rng.normal(0, 0.007)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.007)))
            v = float(np.round(vol_level * vol_mult[i] / raw_mult[i]))
            cur_sym = "STK15N" if (sym == "STK15" and i >= 350) else sym
            cur_isin = "INE103A01029" if (sym == "STK03" and i >= 300) else isin
            if sym == "STK05" and i == 200:
                h = c * 0.99  # bad row: high below close
            if sym == "STK09" and i == 450:
                c = c * 1.45  # bad print, reverts next day
            bhav_rows.append(dict(
                date=d, symbol=cur_sym, series="EQ", isin=cur_isin, open=round(o, 2), high=round(h, 2),
                low=round(lo, 2), close=round(c, 2), last=round(c, 2), prev_close=round(prev, 2), volume=v,
                traded_value=v * (o + h + lo + c) / 4, trades=float(max(1, int(v / 150))), source_file="synthetic",
            ))
            dq = float(np.floor(v * deliv_pct[i] / 100))
            deliv_rows.append(dict(date=d, symbol=cur_sym, series="EQ", traded_qty=v, deliverable_qty=dq,
                                   delivery_pct=round(100 * dq / v, 2) if v else np.nan, source_file="synthetic"))
            for flag, side in ((bulk_buy[i], "BUY"), (bulk_sell[i], "SELL")):
                if flag:
                    for client in ("ALPHA FUND", "BETA CAPITAL LLP"):
                        deal_rows.append(dict(date=d, symbol=cur_sym, client_name=client, side=side,
                                              qty=round(v * 0.05), price=round(c, 2), deal_type="bulk",
                                              source_file="synthetic"))
            if rng.random() < 0.01:  # background noise deals
                deal_rows.append(dict(date=d, symbol=cur_sym, client_name="NOISE TRADER", side=rng.choice(["BUY", "SELL"]),
                                      qty=round(v * 0.02), price=round(c, 2), deal_type="block", source_file="synthetic"))
            if i in div_amt:
                ca_rows.append(dict(symbol=cur_sym, series="EQ", isin=cur_isin, ex_date=d, record_date=d,
                                    subject=f"Dividend - Rs {div_amt[i]:.2f} Per Share", face_value=10.0,
                                    source_file="synthetic"))
        for d, _f, subject, listed in share_events:
            if listed:
                ca_rows.append(dict(symbol=sym, series="EQ", isin=isin, ex_date=days[d], record_date=days[d],
                                    subject=subject, face_value=10.0, source_file="synthetic"))

    bhav = pd.DataFrame(bhav_rows)
    delivery = pd.DataFrame(deliv_rows)
    deals = pd.DataFrame(deal_rows)
    ca = pd.DataFrame(ca_rows)
    nifty = 10000 * np.exp(np.cumsum(mkt))
    index = pd.concat([
        pd.DataFrame(dict(date=days, index_name="Nifty 500", open=nifty, high=nifty, low=nifty, close=nifty)),
        pd.DataFrame(dict(date=days, index_name="Nifty 50", open=nifty, high=nifty, low=nifty, close=nifty)),
    ]).assign(source_file="synthetic")
    calendar = pd.DataFrame(dict(date=days, has_bhavcopy=True, has_index=True, has_delivery=True,
                                 is_trading_day=True, is_gap=False))
    tables = dict(bhavcopy=bhav, delivery=delivery, deals=deals, index=index, corporate_actions=ca, calendar=calendar)
    return tables, truth


def write_fixture(out_dir: Path, seed: int = 7) -> dict[str, int]:
    from smd.validation.schemas import validate

    tables, truth = make_fixture(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in tables.items():
        schema = {"corporate_actions": "bronze.corporate_actions"}.get(name, f"bronze.{name}")
        df = validate(df.reset_index(drop=True), schema)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(df)
    pd.DataFrame(
        [("accumulation", s, d) for s, d in truth.accumulation] + [("distribution", s, d) for s, d in truth.distribution],
        columns=["kind", "symbol", "date"],
    ).to_parquet(out_dir / "truth_events.parquet", index=False)
    return counts
