"""Portfolio construction: selection with a turnover buffer, weighting, and caps.

* Selection: keep a current holding while its score rank <= `sell_rank_buffer * top_k`;
  fill the remaining slots with the best-ranked new names (turnover control).
* Weighting: `equal`, or `hrp` = skfolio HierarchicalRiskParity on trailing daily returns.
* Caps: `max_weight` per stock, and a liquidity cap
  `weight <= max_adv_participation * median_traded_value_21d / equity`. Excess weight is
  redistributed to uncapped names (water-filling); what cannot be placed stays in cash.
* Sector cap: NSE archives carry no point-in-time sector history, so it is not applied
  (DECISIONS.md D-016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smd.utils import get_logger

log = get_logger(__name__)


def select_names(scores: pd.Series, held: set[str], top_k: int, buffer: float) -> list[str]:
    """`scores`: isin -> score for tradeable universe names on the signal date."""
    ranked = scores.sort_values(ascending=False, kind="mergesort")
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    keep = [s for s in ranked.index if s in held and rank[s] <= buffer * top_k][:top_k]
    for s in ranked.index:
        if len(keep) >= top_k:
            break
        if s not in keep:
            keep.append(s)
    return keep


def equal_weights(names: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def hrp_weights(names: list[str], returns: pd.DataFrame) -> pd.Series:
    """skfolio Hierarchical Risk Parity on trailing returns; falls back to equal weight."""
    from skfolio.optimization import HierarchicalRiskParity

    r = returns.reindex(columns=names)
    ok = r.notna().mean() >= 0.6
    good = [n for n in names if ok.get(n, False)]
    if len(good) < 3:
        return equal_weights(names)
    try:
        model = HierarchicalRiskParity()
        model.fit(r[good].fillna(0.0))
        w = pd.Series(model.weights_, index=good)
    except Exception as exc:  # noqa: BLE001 - optimiser failure must not stop a backtest
        log.warning("HRP failed (%s); using equal weights", exc)
        return equal_weights(names)
    missing = [n for n in names if n not in good]
    if missing:  # names without enough history get the average weight
        w = pd.concat([w * len(good) / len(names), pd.Series(1.0 / len(names), index=missing)])
    return w / w.sum()


def apply_caps(w: pd.Series, caps: pd.Series) -> pd.Series:
    """Cap each weight at `caps[name]` and redistribute the excess pro rata to uncapped names."""
    w = w.copy().astype(float)
    caps = caps.reindex(w.index).fillna(np.inf).clip(lower=0.0)
    total = w.sum()
    for _ in range(100):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        free = (~over) & (w < caps - 1e-12)
        if not free.any() or excess <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    assert w.sum() <= total + 1e-9
    return w


def target_weights(names: list[str], weighting: str, returns: pd.DataFrame | None, max_weight: float,
                   liq_caps: pd.Series | None = None) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    w = hrp_weights(names, returns) if weighting == "hrp" and returns is not None else equal_weights(names)
    caps = pd.Series(max_weight, index=w.index)
    if liq_caps is not None:
        caps = np.minimum(caps, liq_caps.reindex(w.index).fillna(0.0))
    return apply_caps(w, caps)
