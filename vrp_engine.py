"""Daily-loop engine (Step 8): settle expiring spreads, mark open spreads, then consider a new entry.

Behavior follows the guide. Two structural changes, neither alters results:
  * build_spread is split into select_spread (strike/expiry/credit; independent of equity and cached
    per date) and size_spread (equity-dependent). The placebo (500 reruns) and Test 5 (45 cells) reuse it.
  * mark_pts reads a per-day {(expiry, strike): (bid, ask)} dict instead of filtering the DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vrp_execution import contracts_for, entry_fees, fill_price
from vrp_selection import pick_expiry, pick_strike


@dataclass
class Spread:
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    k_short: float
    k_long: float
    credit_pts: float
    contracts: float
    fees: float
    max_loss_pts: float
    last_mark: float = 0.0   # cost to close, in points; carried forward if quotes are missing


@dataclass(frozen=True)
class Pick:
    expiry: pd.Timestamp
    k_short: float
    k_long: float
    credit_pts: float
    max_loss_pts: float


def select_spread(chain: pd.DataFrame, d, cfg: dict, f: float):
    if chain is None or len(chain) == 0:
        return None
    exp = pick_expiry(chain, d, cfg["target_dte"])
    if exp is None:
        return None
    if abs((exp - d).days - cfg["target_dte"]) > cfg["dte_tolerance"]:
        return None
    puts = chain[(chain["expiry"] == exp) & (chain["cp"] == "P") & (chain["ask"] > 0)]
    shorts = puts[puts["bid"] > 0]
    if puts.empty or shorts.empty:                       # guard (not in the guide): argmin of empty raises
        return None
    short = pick_strike(shorts, -cfg["short_delta"])
    long_ = pick_strike(puts, -cfg["long_delta"])
    if long_["strike"] >= short["strike"]:
        return None
    credit = fill_price(short, "sell", f) - fill_price(long_, "buy", f)
    max_loss = (short["strike"] - long_["strike"]) - credit
    if credit <= 0 or max_loss <= 0:
        return None
    return Pick(exp, float(short["strike"]), float(long_["strike"]), float(credit), float(max_loss))


def size_spread(pick: Pick, d, equity: float, cfg: dict):
    if pick is None or equity <= 0:
        return None
    contracts = contracts_for(equity, pick.max_loss_pts, cfg)
    return Spread(d, pick.expiry, pick.k_short, pick.k_long, pick.credit_pts, contracts,
                  entry_fees(contracts, cfg), pick.max_loss_pts, last_mark=pick.credit_pts)


def build_spread(chain, d, equity, cfg, f):
    """Guide-compatible wrapper."""
    return size_spread(select_spread(chain, d, cfg, f), d, equity, cfg)


def quote_book(chain: pd.DataFrame) -> dict:
    """{(expiry, strike): (bid, ask)} for the day's puts."""
    p = chain[chain["cp"] == "P"]
    return dict(zip(zip(p["expiry"], p["strike"]), zip(p["bid"], p["ask"])))


def mark_pts(p: Spread, quotes: dict, mode: str = "mid") -> float:
    """Cost to close in points. mode='mid' (guide default) or 'conservative' (short at ask, long at bid)."""
    qs, ql = quotes.get((p.expiry, p.k_short)), quotes.get((p.expiry, p.k_long))
    if qs is None or ql is None:
        return p.last_mark
    p.last_mark = (qs[1] - ql[0]) if mode == "conservative" else (sum(qs) - sum(ql)) / 2
    return p.last_mark


def run_backtest(days, chain_for, settle, signal, cfg, f, cache=None, mark="mid"):
    """days: DatetimeIndex of trading days.
    chain_for(d): DataFrame of that day's option quotes.
    settle: Series mapping expiry -> official settlement value.
    signal: boolean Series, already lagged (Step 5).
    cache: optional dict shared across runs that use the same chain_for (speeds up placebo/grid runs)."""
    cache = {} if cache is None else cache
    qcache, pcache = cache.setdefault("quotes", {}), cache.setdefault("picks", {})
    periods = days.to_period("W")
    is_first = np.r_[True, np.asarray(periods[1:] != periods[:-1])]       # first trading day of each week
    sig = signal.reindex(days, fill_value=False).to_numpy(dtype=bool)
    start = cfg["start_equity"]
    realized, open_pos, curve, trades = 0.0, [], {}, []

    for i, d in enumerate(days):
        # 1. Settle expiring spreads at the official settlement value.
        for p in [p for p in open_pos if p.expiry <= d]:
            if p.expiry not in settle.index:
                raise KeyError(f"no official settlement value for expiry {p.expiry.date()}")
            S = settle.loc[p.expiry]
            pnl_pts = p.credit_pts - max(p.k_short - S, 0) + max(p.k_long - S, 0)
            realized += pnl_pts * 100 * p.contracts
            trades.append((p.open_date, p.expiry, p.k_short, p.k_long, p.credit_pts,
                           p.max_loss_pts, pnl_pts, p.contracts, p.fees))
            open_pos.remove(p)

        # 2. Mark open spreads; equity before any new entry is used for sizing.
        if open_pos:
            if d not in qcache:
                qcache[d] = quote_book(chain_for(d))
            for p in open_pos:
                mark_pts(p, qcache[d], mark)
        mtm = sum((p.credit_pts - p.last_mark) * 100 * p.contracts for p in open_pos)
        equity = start + realized + mtm

        # 3. Enter on the first trading day of the week if the gate is on and capacity remains.
        if is_first[i] and sig[i] and len(open_pos) < cfg["max_open"]:
            key = (d, f, cfg["target_dte"], cfg["short_delta"], cfg["long_delta"])
            if key not in pcache:
                pcache[key] = select_spread(chain_for(d), d, cfg, f)
            p = size_spread(pcache[key], d, equity, cfg)
            if p is not None:
                realized -= p.fees
                open_pos.append(p)

        # 4. Record end-of-day equity.
        mtm = sum((p.credit_pts - p.last_mark) * 100 * p.contracts for p in open_pos)
        curve[d] = start + realized + mtm
    return pd.Series(curve), pd.DataFrame(
        trades, columns=["open", "expiry", "k_short", "k_long", "credit_pts",
                         "max_loss_pts", "pnl_pts", "contracts", "fees"])
