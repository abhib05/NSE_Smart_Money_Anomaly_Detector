"""Corporate-action parsing and backward adjustment factors.

Price factor of one event (applied to every price *before* the ex-date):

* split  face value 10 -> 2              : 2/10 = 0.2
* consolidation  face value 1 -> 10      : 10
* bonus  a:b (a new shares per b held)   : b / (a + b)   (1:1 -> 0.5, 1:2 -> 2/3)

`adj_factor(t) = prod(price_factor of events with ex_date > t)`;
`adj_price = raw_price * adj_factor`, `adj_volume = raw_volume / adj_factor`.

Dividends feed a separate total-return factor: on the ex-date the event factor is
`1 - dividend / prev_close`, and `tr_factor(t) = adj_factor(t) * prod(div factors after t)`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pandas as pd

from smd.utils import get_logger

log = get_logger(__name__)

_RS = r"R(?:s|e)\.?\s*"
_AMT_RE = re.compile(r"R(?:s|e)\.?\s*([\d]+(?:\.\d+)?)", re.I)
_SPLIT_WORDS = ("split", "splt", "sub-division", "subdivision", "sub division", "consolidat")
_BONUS_RE = re.compile(r"bonus\s*(?:issue)?\s*[-:]?\s*(\d+)\s*:\s*(\d+)", re.I)
_RIGHTS_RE = re.compile(r"rights\s*(?:issue)?\s*[-:]?\s*(\d+)\s*:\s*(\d+)", re.I)
_DIV_AMT_RE = re.compile(rf"{_RS}([\d]+(?:\.\d+)?)", re.I)
_DIV_PCT_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*%", re.I)

CA_COLUMNS = ["isin", "symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "cash_amount", "price_factor", "subject"]


def parse_subject(subject: str, face_value: float | None = None) -> list[dict]:
    """Parse NSE's free-text corporate action `subject` into zero or more typed events."""
    s = " ".join(str(subject).split())
    low = s.lower()
    events: list[dict] = []
    if any(w in low for w in _SPLIT_WORDS):
        # "From Rs 10/- To Re 1/-", "Rs.10/- To Re.1/-", "Fv Splt Frm Rs 10 To Re 1": first two amounts
        start = min(low.find(w) for w in _SPLIT_WORDS if w in low)  # skip e.g. a dividend amount before it
        amts = _AMT_RE.findall(s[start:])
        if len(amts) >= 2:
            old, new = float(amts[0]), float(amts[1])
            if old > 0 and new > 0 and old != new:
                kind = "split" if new < old else "consolidation"
                events.append(dict(action_type=kind, ratio_num=old, ratio_den=new, cash_amount=0.0,
                                   price_factor=new / old))
    if "bonus" in low:
        m = _BONUS_RE.search(s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 0 and b > 0:
                events.append(dict(action_type="bonus", ratio_num=a, ratio_den=b, cash_amount=0.0,
                                   price_factor=b / (a + b)))
    if "rights" in low:
        m = _RIGHTS_RE.search(s)
        events.append(dict(action_type="rights", ratio_num=float(m.group(1)) if m else np.nan,
                           ratio_den=float(m.group(2)) if m else np.nan, cash_amount=0.0, price_factor=1.0))
    if "dividend" in low:
        amounts = [float(x) for x in _DIV_AMT_RE.findall(s)]
        cash = sum(amounts)
        if not amounts and face_value:
            pcts = [float(x) for x in _DIV_PCT_RE.findall(s)]
            cash = sum(p / 100.0 * face_value for p in pcts)
        if cash > 0:
            events.append(dict(action_type="dividend", ratio_num=np.nan, ratio_den=np.nan,
                               cash_amount=cash, price_factor=1.0))
    if "amalgamation" in low or "merger" in low or "scheme of arrangement" in low:
        events.append(dict(action_type="merger", ratio_num=np.nan, ratio_den=np.nan, cash_amount=0.0,
                           price_factor=1.0))
    return events


def parse_corporate_actions(bronze_ca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in bronze_ca.itertuples(index=False):
        if pd.isna(rec.ex_date):
            continue
        for ev in parse_subject(rec.subject, rec.face_value if not pd.isna(rec.face_value) else None):
            rows.append(dict(isin=rec.isin, symbol=rec.symbol, ex_date=rec.ex_date, subject=rec.subject, **ev))
    out = pd.DataFrame(rows, columns=CA_COLUMNS)
    out = out.drop_duplicates(["symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "cash_amount"])
    return out.sort_values(["ex_date", "symbol"]).reset_index(drop=True)


def typical_split_factors() -> list[float]:
    """Price factors of common splits, bonuses and consolidations (for jump matching)."""
    f: set[Fraction] = {Fraction(1, n) for n in (2, 3, 4, 5, 10)} | {Fraction(2, 5), Fraction(2, 10)}
    f |= {Fraction(b, a + b) for a in range(1, 6) for b in range(1, 6)}
    f |= {Fraction(n) for n in (2, 5, 10)}
    return sorted(float(x) for x in f if abs(float(x) - 1) > 0.15)


def match_typical_factor(ratio: float, tol: float) -> float | None:
    """Return the typical factor within relative tolerance `tol` of `ratio`, if any."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    best = min(typical_split_factors(), key=lambda f: abs(np.log(ratio / f)))
    return best if abs(ratio / best - 1) <= tol else None


def event_factors(panel: pd.DataFrame, ca: pd.DataFrame, infer_from_prevclose: bool, tol: float) -> pd.DataFrame:
    """Per-row event factors on ex-dates.

    `panel` must be sorted by (isin, date) with columns isin, date, close, prev_close.
    `ca` must carry a `sid` column (mapped security id). Returns `panel` with added columns
    `split_event` (price factor applied on that ex-date, 1 otherwise), `div_cash`,
    `ca_event` (description) and the boolean flag `ca_inferred_from_prevclose`.
    """
    p = panel
    # ex-date -> first trading day of that security on/after the ex-date
    pos_ca = ca.dropna(subset=["sid"]).copy()
    pos_ca["ex_date"] = pd.to_datetime(pos_ca["ex_date"])
    idx = p[["isin", "date"]].reset_index()
    if pos_ca.empty:
        m = pd.DataFrame(columns=["index", "action_type", "price_factor", "cash_amount"])
    else:
        m = pd.merge_asof(
            pos_ca.sort_values("ex_date"), idx.sort_values("date").rename(columns={"isin": "sid"}),
            left_on="ex_date", right_on="date", by="sid", direction="forward",
            tolerance=pd.Timedelta(days=10),
        ).dropna(subset=["index"])
    m["index"] = m["index"].astype(int)

    split_event = pd.Series(1.0, index=p.index)
    div_cash = pd.Series(0.0, index=p.index)
    desc = pd.Series("", index=p.index, dtype=object)
    px = m[m["action_type"].isin(["split", "bonus", "consolidation"])]
    for i, grp in px.groupby("index"):
        split_event.loc[i] = float(np.prod(grp["price_factor"].to_numpy()))
        desc.loc[i] = ";".join(grp["action_type"] + ":" + grp["price_factor"].round(4).astype(str))
    dv = m[m["action_type"] == "dividend"].groupby("index")["cash_amount"].sum()
    div_cash.loc[dv.index] = dv.to_numpy()
    for i in dv.index:
        desc.loc[i] = (desc.loc[i] + ";" if desc.loc[i] else "") + f"dividend:{dv.loc[i]:.2f}"

    prev_close_raw = p.groupby("isin", sort=False)["close"].shift(1)
    implied = p["prev_close"] / prev_close_raw
    has_px_event = split_event != 1.0
    consecutive = p["date"].map(pd.Series(np.arange(len(p["date"].unique())), index=np.sort(p["date"].unique())))
    consecutive = consecutive.groupby(p["isin"], sort=False).diff() == 1
    inferred = pd.Series(False, index=p.index)
    if infer_from_prevclose:
        # only on consecutive trading days: after a gap prev_close is stale, not adjusted
        cand = (~has_px_event) & consecutive & implied.notna() & ((implied - 1).abs() > 0.15)
        for i in cand[cand].index:
            f = match_typical_factor(float(implied.loc[i]), tol)
            if f is not None:
                split_event.loc[i] = f
                inferred.loc[i] = True
                desc.loc[i] = (desc.loc[i] + ";" if desc.loc[i] else "") + f"inferred:{f:.4f}"
    out = p  # columns are added in place (memory: the full panel is ~5M rows)
    out["split_event"] = split_event
    out["div_cash"] = div_cash
    out["ca_event"] = desc
    out["ca_inferred_from_prevclose"] = inferred
    log.info(
        "corporate actions: %d price events applied (%d inferred from prev_close), %d dividend events",
        int((split_event != 1).sum()), int(inferred.sum()), int((div_cash > 0).sum()),
    )
    return out


def _backward_cumprod(event: pd.Series, groups: pd.Series) -> pd.Series:
    """factor(t) = prod(event_s for s > t within group)."""
    rev = event.iloc[::-1]
    cum = rev.groupby(groups.iloc[::-1], sort=False).cumprod().iloc[::-1]  # prod over s >= t
    return cum.groupby(groups, sort=False).shift(-1).fillna(1.0)  # prod over s > t


def adjustment_factors(p: pd.DataFrame) -> pd.DataFrame:
    """Adds adj_factor, tr_factor and adjusted columns. `p` sorted by (isin, date) with event columns."""
    out = p
    out["adj_factor"] = _backward_cumprod(out["split_event"], out["isin"])
    prev_close = out["prev_close"].where(out["prev_close"] > 0, out.groupby("isin", sort=False)["close"].shift(1))
    div_ev = (1.0 - out["div_cash"] / prev_close).where(out["div_cash"] > 0, 1.0)
    out["div_too_large"] = div_ev < 0.5
    div_ev = div_ev.clip(lower=0.5, upper=1.0).fillna(1.0)
    out["tr_factor"] = out["adj_factor"] * _backward_cumprod(div_ev, out["isin"])
    for c in ("open", "high", "low", "close"):
        out[f"adj_{c}"] = out[c] * out["adj_factor"]
    out["adj_open_tr"] = out["open"] * out["tr_factor"]
    out["adj_close_tr"] = out["close"] * out["tr_factor"]
    out["adj_volume"] = out["volume"] / out["adj_factor"]
    out["adj_deliverable_qty"] = out["deliverable_qty"] / out["adj_factor"]
    return out
