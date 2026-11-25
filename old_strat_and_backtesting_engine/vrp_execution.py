"""Fills, fees, sizing (Step 7)."""
from __future__ import annotations

import pandas as pd


def fill_price(row: pd.Series, side: str, f: float) -> float:
    """Pay fraction f of the half-spread away from mid. f = 1.0 sells at the bid and buys at the ask."""
    mid = (row["bid"] + row["ask"]) / 2
    half = (row["ask"] - row["bid"]) / 2
    return mid - f * half if side == "sell" else mid + f * half


def contracts_for(equity: float, max_loss_pts: float, cfg: dict) -> float:
    """Risk risk_per_trade of current equity per spread. Fractional contracts are allowed in the backtest."""
    return cfg["risk_per_trade"] * equity / (max_loss_pts * 100)


def entry_fees(contracts: float, cfg: dict) -> float:
    """fee_per_contract per leg at entry; cash settlement means no closing fee."""
    return 2 * contracts * cfg["fee_per_contract"]
