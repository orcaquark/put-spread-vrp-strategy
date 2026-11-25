"""HAR-RV forecast (Step 4) and the VRP / term-structure gate (Step 5).

Named signals.py, not signal.py: a module called signal.py shadows the standard-library `signal`
module whenever src/ lands on sys.path (PYTHONPATH=src, `cd src && python ...`, some IDE run configs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def har_features(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"d": rv, "w": rv.rolling(5).mean(), "m": rv.rolling(22).mean()})


def forward_mean(rv: pd.Series, h: int = 22) -> pd.Series:
    """Mean of rv over t+1..t+h, aligned to t."""
    return rv[::-1].rolling(h).mean()[::-1].shift(-1)


def har_forecast(rv: pd.Series, refit_every: int = 21, min_obs: int = 750, h: int = 22) -> pd.Series:
    """Forecast of mean daily variance over the next h trading days, made at the close of each day."""
    X = har_features(rv)
    y = forward_mean(rv, h).rename("y")
    data = pd.concat([X, y], axis=1)
    out = pd.Series(np.nan, index=rv.index)
    beta = None
    for i in range(min_obs, len(rv)):
        if beta is None or (i - min_obs) % refit_every == 0:
            # Use only rows whose h-day target window has fully ended by day i (the lookahead guard).
            train = data.iloc[: i - h + 1].dropna()
            fit = sm.OLS(train["y"], sm.add_constant(train[["d", "w", "m"]])).fit()
            beta = fit.params
        x = X.iloc[i]
        if x.isna().any():
            continue
        out.iloc[i] = beta["const"] + beta["d"] * x["d"] + beta["w"] * x["w"] + beta["m"] * x["m"]
    return out


def build_signals(vix: pd.Series, vix3m: pd.Series, rv_forecast: pd.Series, cfg: dict) -> dict:
    """Returns boolean Series (already lagged one day) for the gated strategy and the benchmarks.

    Also returns 'first_tradable': the first date on which the gated signal can be True.
    """
    vrp = (vix / 100) ** 2 - 252 * rv_forecast                      # annualized variance units
    threshold = vrp.expanding(min_periods=252).median().shift(1)    # today's VRP never sets its own cutoff
    gate_vrp = vrp > threshold
    gate_ts = (vix / vix3m) < cfg["gate"]["term_structure_max"]
    lag = lambda s: s.shift(1, fill_value=False).astype(bool)       # compute at close t-1, trade at t
    first_valid = threshold.first_valid_index()
    pos = vrp.index.get_loc(first_valid)
    first_tradable = vrp.index[pos + 1] if pos + 1 < len(vrp) else None
    return {"gated": lag(gate_vrp & gate_ts), "vrp_only": lag(gate_vrp), "ts_only": lag(gate_ts),
            "ungated": pd.Series(True, index=vix.index), "vrp": vrp, "threshold": threshold,
            "first_tradable": first_tradable}
