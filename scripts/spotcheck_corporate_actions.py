"""Phase 2 acceptance: manual spot-check of known splits/bonuses on real data.

Picks the most liquid securities that had a split or bonus during the study period (plus
any symbols passed on the command line), prints raw vs adjusted prices around each ex-date
and the adjusted return on the ex-date, and writes a Markdown table for docs/DATA.md.

Usage: python scripts/spotcheck_corporate_actions.py [--n 5] [SYMBOL ...]
"""

from __future__ import annotations

import argparse

import pandas as pd

from smd.config import load_config
from smd.utils import read_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    panel = read_dataset(cfg, "silver", "daily_panel",
                         columns=["date", "isin", "symbol", "close", "prev_close", "adj_close", "adj_close_tr",
                                  "traded_value", "ca_event", "dq_flags"])
    panel = panel.sort_values(["isin", "date"])
    panel["adj_ret"] = panel.groupby("isin")["adj_close"].pct_change()
    panel["raw_ret"] = panel.groupby("isin")["close"].pct_change()
    ev = panel[panel["ca_event"].str.contains("split|bonus|inferred", na=False)]
    ev = ev[ev["date"] >= pd.Timestamp(cfg.study_start)]
    liq = panel.groupby("isin")["traded_value"].median()
    ev = ev.assign(liq=ev["isin"].map(liq)).sort_values("liq", ascending=False)
    picked = ev[ev["symbol"].isin(args.symbols)] if args.symbols else ev.drop_duplicates("isin").head(args.n)
    rows = []
    for e in picked.itertuples():
        s = panel[panel["isin"] == e.isin].set_index("date")
        win = s.loc[: e.date].tail(3).index.union(s.loc[e.date:].head(3).index)
        print(f"\n== {e.symbol} ex-date {e.date:%Y-%m-%d}: {e.ca_event}")
        print(s.loc[win, ["symbol", "close", "prev_close", "adj_close", "raw_ret", "adj_ret", "dq_flags"]].to_string())
        rows.append({"symbol": e.symbol, "ex_date": f"{e.date:%Y-%m-%d}", "event": e.ca_event,
                     "raw_return_on_ex_date": f"{e.raw_ret:+.1%}", "adjusted_return_on_ex_date": f"{e.adj_ret:+.2%}"})
    out = pd.DataFrame(rows)
    print("\n| " + " | ".join(out.columns) + " |\n|" + "---|" * len(out.columns))
    for r in out.itertuples(index=False):
        print("| " + " | ".join(map(str, r)) + " |")


if __name__ == "__main__":
    main()
