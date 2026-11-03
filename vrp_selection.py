"""Expiry and strike selection (Step 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def pick_expiry(chain: pd.DataFrame, d: pd.Timestamp, target_dte: int = 30):
    """Expiry whose DTE is closest to target_dte. Returns None if the chain has no future expiry."""
    exp = chain["expiry"].drop_duplicates().reset_index(drop=True)
    exp = exp[exp > d].reset_index(drop=True)          # guard (not in the guide): never pick an expired date
    if exp.empty:
        return None
    dte = (exp - d).dt.days
    return exp.iloc[(dte - target_dte).abs().argmin()]


def pick_strike(puts: pd.DataFrame, target_delta: float) -> pd.Series:
    """target_delta is negative for puts, for example -0.20. `puts` must be non-empty."""
    return puts.iloc[(puts["delta"] - target_delta).abs().argmin()]


def bs_put_delta(F: float, K: float, T: float, iv: float, r: float) -> float:
    """Black-76 put delta on the forward. Use only if the vendor supplies no delta."""
    d1 = (np.log(F / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    return -np.exp(-r * T) * norm.cdf(-d1)
