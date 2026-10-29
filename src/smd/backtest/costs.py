"""Indian equity-delivery transaction costs + slippage (rates in `configs/costs.yaml`).

Buy:  STT + stamp duty + exchange txn + SEBI fee + brokerage + GST(brokerage+exchange+SEBI) + slippage
Sell: STT + exchange txn + SEBI fee + brokerage + GST(...) + DP charge (per scrip) + slippage
Slippage (bps) = base_bps + impact_bps_at_1pct * sqrt(participation / 1%), participation =
trade value / that day's traded value.
"""

from __future__ import annotations

import numpy as np

from smd.config import CostsCfg


def slippage_bps(trade_value: float, day_traded_value: float, cfg: CostsCfg) -> float:
    part = trade_value / day_traded_value if day_traded_value and day_traded_value > 0 else 0.01
    return cfg.slippage.base_bps + cfg.slippage.impact_bps_at_1pct * float(np.sqrt(max(part, 0.0) / 0.01))


def trade_cost(value: float, side: str, day_traded_value: float, cfg: CostsCfg) -> dict[str, float]:
    """Cost components (₹, all >= 0) for one order of absolute value `value`."""
    v = abs(float(value))
    if v == 0:
        return {"stt": 0.0, "stamp": 0.0, "exchange": 0.0, "sebi": 0.0, "brokerage": 0.0, "gst": 0.0, "dp": 0.0,
                "slippage": 0.0}
    brokerage = min(v * cfg.brokerage_pct, cfg.brokerage_cap_inr) if cfg.brokerage_pct > 0 else 0.0
    exchange = v * cfg.exchange_txn
    sebi = v * cfg.sebi_fee
    out = {
        "stt": v * (cfg.stt_buy if side == "buy" else cfg.stt_sell),
        "stamp": v * cfg.stamp_duty_buy if side == "buy" else 0.0,
        "exchange": exchange,
        "sebi": sebi,
        "brokerage": brokerage,
        "gst": cfg.gst_rate * (brokerage + exchange + sebi),
        "dp": cfg.dp_charge_per_sell_inr if side == "sell" else 0.0,
        "slippage": v * slippage_bps(v, day_traded_value, cfg) / 1e4,
    }
    return out


def round_trip_cost_bps(value: float, day_traded_value: float, cfg: CostsCfg) -> float:
    b = sum(trade_cost(value, "buy", day_traded_value, cfg).values())
    s = sum(trade_cost(value, "sell", day_traded_value, cfg).values())
    return (b + s) / value * 1e4
