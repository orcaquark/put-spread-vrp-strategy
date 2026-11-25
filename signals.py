"""Trend and relative-strength signals for a multi-asset long-only book.

All functions read a price DataFrame (rows = completed daily closes, columns = tickers)
and return a value per ticker computed from the final row. No QuantConnect dependency,
so every function can be tested offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def ewma_vol_ann(returns: pd.DataFrame, lam: float = 0.94, min_obs: int = 40) -> pd.Series:
    """Annualized EWMA volatility per column, from zero-mean squared returns."""
    var = (
        returns.pow(2)
        .ewm(alpha=1.0 - lam, adjust=False, min_periods=min_obs)
        .mean()
        .iloc[-1]
    )
    return np.sqrt(var * TRADING_DAYS)


def tsmom_score(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    lam: float = 0.94,
    k: float = 2.0,
) -> pd.Series:
    """Time-series momentum in [0, 1].

    For each lookback L: z = L-day return / (daily vol * sqrt(L)); score = tanh(k * z).
    Scores average across lookbacks; negatives clip to zero (long-only).
    Assets with fewer than two usable lookbacks score zero.
    """
    daily_vol = ewma_vol_ann(log_returns(prices), lam) / np.sqrt(TRADING_DAYS)
    last = prices.iloc[-1]
    scores = pd.DataFrame(np.nan, index=prices.columns, columns=list(lookbacks))

    for n in lookbacks:
        if len(prices) > n:
            ret = last / prices.iloc[-1 - n] - 1.0
            z = ret / (daily_vol * np.sqrt(n))
            scores[n] = np.tanh(k * z)

    mean = scores.mean(axis=1, skipna=True)
    mean[scores.notna().sum(axis=1) < 2] = 0.0

    return mean.fillna(0.0).clip(lower=0.0)


def ma_score(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (50, 100, 200),
) -> pd.Series:
    """Fraction of moving averages (including today's close) that price sits above."""
    last = prices.iloc[-1]
    flags = pd.DataFrame(np.nan, index=prices.columns, columns=list(windows))

    for n in windows:
        if len(prices) >= n:
            ma = prices.iloc[-n:].mean()
            flags[n] = (last > ma).astype(float)

    out = flags.mean(axis=1, skipna=True)
    out[flags.notna().sum(axis=1) < 2] = 0.0

    return out.fillna(0.0)


def relative_strength_tilt(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    lam: float = 0.94,
    lo: float = 0.7,
    hi: float = 1.3,
) -> pd.Series:
    """Cross-sectional momentum tilt in [lo, hi].

    Ranks vol-adjusted return over [t-skip-lookback, t-skip]. Skipping the latest month
    avoids short-term reversal. Assets without enough history get a neutral tilt of 1.0.
    """
    tilt = pd.Series(1.0, index=prices.columns)

    if len(prices) <= lookback + skip:
        return tilt

    ann_vol = ewma_vol_ann(log_returns(prices), lam)
    ret = prices.iloc[-1 - skip] / prices.iloc[-1 - skip - lookback] - 1.0
    adj = (ret / ann_vol).replace([np.inf, -np.inf], np.nan).dropna()

    if len(adj) < 3:
        return tilt

    pct = adj.rank(pct=True)
    pct = (pct - pct.min()) / max(pct.max() - pct.min(), 1e-9)
    tilt.loc[pct.index] = lo + (hi - lo) * pct

    return tilt


def trend_signal(
    prices: pd.DataFrame,
    w_tsmom: float = 0.5,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    ma_windows: tuple[int, ...] = (50, 100, 200),
    use_tilt: bool = True,
    tilt_lookback: int = 126,
    tilt_skip: int = 21,
    tilt_lo: float = 0.7,
    tilt_hi: float = 1.3,
    cap: float = 1.3,
) -> pd.Series:
    """Blend momentum and MA trend, then apply the relative-strength tilt.

    Trend alone is in [0, 1]. The tilt scales it by [tilt_lo, tilt_hi]. A zero trend
    score always stays zero, so a tilt never opens a position in a downtrend.
    """
    trend = (
        w_tsmom * tsmom_score(prices, lookbacks)
        + (1.0 - w_tsmom) * ma_score(prices, ma_windows)
    )

    if use_tilt:
        trend = trend * relative_strength_tilt(
            prices, tilt_lookback, tilt_skip, lo=tilt_lo, hi=tilt_hi
        )

    return trend.clip(lower=0.0, upper=cap)