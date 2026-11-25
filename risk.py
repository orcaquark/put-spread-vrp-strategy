"""Covariance estimation and risk-budgeted weight construction.

Framework: risk budgeting with a covariance-aware volatility target.

    reference weights   ref_i = budget_i / vol_i               (full-signal risk parity)
    scale               k = target_vol / max(vol_fast(ref), vol_slow(ref))
    final weights       w_i = k * signal_i * ref_i

Because k is computed at full signal, weak trend signals reduce absolute risk instead of
being re-levered to the target. The max of a fast and a slow covariance estimate makes the
target react within days to a volatility spike and decay slowly afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_cov(returns: pd.DataFrame, lam: float, min_obs: int = 60) -> pd.DataFrame | None:
    """Annualized zero-mean EWMA covariance on the common (no-NaN) sample."""
    x = returns.dropna(how="any")

    if len(x) < min_obs:
        return None

    n = len(x)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    xv = x.to_numpy(dtype=float)
    cov = (xv * w[:, None]).T @ xv * TRADING_DAYS

    return pd.DataFrame(cov, index=x.columns, columns=x.columns)


def shrink_to_diag(cov: pd.DataFrame, delta: float = 0.2) -> pd.DataFrame:
    """Shrink off-diagonal covariance toward zero by a constant factor."""
    diag = np.diag(np.diag(cov.to_numpy()))
    out = (1.0 - delta) * cov.to_numpy() + delta * diag

    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    v = w.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)

    return float(np.sqrt(max(v @ cov.to_numpy() @ v, 0.0)))


def class_budgets(
    tickers: list[str],
    asset_class: dict[str, str],
    class_budget: dict[str, float],
) -> pd.Series:
    """Equal split inside each asset class; classes with no valid asset drop out; sums to 1."""
    counts: dict[str, int] = {}

    for t in tickers:
        counts[asset_class[t]] = counts.get(asset_class[t], 0) + 1

    raw = pd.Series(
        {t: class_budget[asset_class[t]] / counts[asset_class[t]] for t in tickers},
        dtype=float,
    )
    total = raw.sum()

    return raw / total if total > 0 else raw


def risk_budget_weights(
    signal: pd.Series,
    vol: pd.Series,
    budgets: pd.Series,
    cov_fast: pd.DataFrame,
    cov_slow: pd.DataFrame,
    target_vol: float,
    max_gross: float,
    max_asset: float,
) -> tuple[pd.Series, dict]:
    """Return long-only weights (fraction of equity) and diagnostics."""
    idx = [
        t for t in signal.index
        if t in vol.index and t in cov_fast.index and t in cov_slow.index
        and np.isfinite(vol[t]) and vol[t] > 0
    ]
    zero = pd.Series(0.0, index=signal.index)

    if not idx:
        return zero, {"k": 0.0, "ref_vol": 0.0, "gross": 0.0}

    ref = budgets.reindex(idx).fillna(0.0) / vol[idx]
    ref_vol = max(portfolio_vol(ref, cov_fast), portfolio_vol(ref, cov_slow))

    if ref_vol <= 0:
        return zero, {"k": 0.0, "ref_vol": 0.0, "gross": 0.0}

    k = target_vol / ref_vol
    w = (k * signal[idx] * ref).clip(lower=0.0, upper=max_asset)
    gross = float(w.sum())

    if gross > max_gross:
        w = w * (max_gross / gross)
        gross = max_gross

    out = zero.copy()
    out.loc[w.index] = w

    return out, {"k": float(k), "ref_vol": float(ref_vol), "gross": gross}