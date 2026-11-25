"""Performance, per-trade metrics, inference (Step 10), break-even fill fraction (Test 4), trial log."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap
from scipy.stats import norm


def perf(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    dd = equity / equity.cummax() - 1
    try:
        month_end = equity.resample("ME").last()         # pandas >= 2.2
    except ValueError:
        month_end = equity.resample("M").last()          # older pandas (QuantConnect's image may be older)
    monthly = month_end.pct_change().dropna()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    return {
        "CAGR": (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 and equity.iloc[-1] > 0 else np.nan,
        "ann_vol": r.std() * np.sqrt(252),
        "sortino": r.mean() * 252 / downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "CVaR_5pct_daily": r[r <= r.quantile(0.05)].mean(),
        "worst_month": monthly.min() if len(monthly) else np.nan,
        "skew": r.skew(),
        "excess_kurtosis": r.kurt(),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    pnl = trades["pnl_pts"]
    ror = pnl / trades["max_loss_pts"]       # return on the capital each spread puts at risk
    return {
        "n_trades": len(trades),
        "avg_pnl_pts": pnl.mean(),
        "avg_return_on_risk": ror.mean(),
        "loss_frequency": (pnl < 0).mean(),
        "worst_trade_pts": pnl.min(),
        "worst_trade_ror": ror.min(),        # -1.0 means a full maximum loss
    }


def alpha_beta(r: pd.Series, mkt_ret: pd.Series, rf_daily: pd.Series | None = None, lags: int = 22) -> dict:
    """OLS of strategy returns on SPX returns with Newey-West (HAC) errors.

    r is option P&L over equity and earns no interest on cash, so it is already an excess return;
    rf is therefore subtracted from the SPX series only (see IMPLEMENTATION_NOTES.md).
    """
    x = mkt_ret.reindex(r.index)
    if rf_daily is not None:
        x = x - rf_daily.reindex(r.index)
    df = pd.concat({"y": r, "x": x}, axis=1).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_annual": fit.params["const"] * 252, "alpha_t": fit.tvalues["const"],
            "p_alpha": fit.pvalues["const"], "beta": fit.params["x"], "n": len(df)}


def bootstrap_mean_ci(r: pd.Series, block: int = 22, reps: int = 2000, seed: int = 0):
    """Stationary block bootstrap 95% CI for the annualized mean return (as in the guide)."""
    bs = StationaryBootstrap(block, r.values, seed=seed)
    lo, hi = bs.conf_int(lambda x: x.mean() * 252, reps=reps, size=0.95).ravel()
    return float(lo), float(hi)


def bootstrap_alpha_ci(r: pd.Series, mkt_ret: pd.Series, rf_daily=None, block=22, reps=2000, seed=0):
    """ADDED: paired stationary block bootstrap CI for annualized alpha, so kill criterion 5's
    'alpha ... under the block bootstrap' is tested on alpha rather than on the mean return."""
    x = mkt_ret.reindex(r.index)
    if rf_daily is not None:
        x = x - rf_daily.reindex(r.index)
    df = pd.concat({"y": r, "x": x}, axis=1).dropna()
    bs = StationaryBootstrap(block, df["y"].values, df["x"].values, seed=seed)
    lo, hi = bs.conf_int(lambda y, x: np.polyfit(x, y, 1)[1] * 252, reps=reps, size=0.95).ravel()
    return float(lo), float(hi)


def sharpe_daily(r: pd.Series) -> float:
    s = r.std()
    return float(r.mean() / s) if s and s > 0 else float("nan")


def deflated_sharpe(sr, n_obs, skew, kurt, n_trials, sr_var):
    """Bailey & Lopez de Prado (2014).
    sr: per-period (non-annualized) Sharpe of the chosen variant. kurt: RAW kurtosis (not excess).
    sr_var: variance of per-period Sharpe ratios across trials. n_trials: rows in trial_log.csv.
    Guard (not in the guide): with <2 trials or an undefined sr_var there is nothing to deflate by,
    so the result is the plain probabilistic Sharpe ratio against 0."""
    g = 0.5772156649
    if n_trials is None or n_trials < 2 or sr_var is None or not np.isfinite(sr_var) or sr_var <= 0:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(sr_var) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                                 + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    radicand = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if radicand <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(radicand)))


def breakeven_fill_fraction(net_pnl, grid=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)) -> float:
    """net_pnl(f) -> final net P&L in dollars at fill fraction f. Returns the f where it crosses zero."""
    pnl = [net_pnl(f) for f in grid]
    if pnl[0] <= 0:
        return 0.0                  # no edge even at the mid price
    for (f0, p0), (f1, p1) in zip(zip(grid, pnl), zip(grid[1:], pnl[1:])):
        if p1 <= 0:
            return f0 + (f1 - f0) * p0 / (p0 - p1)    # linear interpolation
    return float("inf")             # still profitable at the largest f in the grid


class TrialLog:
    """trial_log.csv: one row per variant, registered BEFORE it is run (guide Step 1).

    Re-registering an identical (description, params) returns the existing id, so reruns of the same
    variant do not inflate the trial count."""
    COLS = ["trial_id", "timestamp", "stage", "description", "params", "sharpe_daily"]

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=self.COLS)

    def _save(self):
        self.df.to_csv(self.path, index=False)

    def register(self, description: str, params: dict, stage: str = "insample") -> int:
        p = json.dumps(params, sort_keys=True)
        hit = self.df[(self.df["description"] == description) & (self.df["params"] == p)]
        if len(hit):
            return int(hit["trial_id"].iloc[0])
        tid = int(self.df["trial_id"].max() + 1) if len(self.df) else 1
        row = {"trial_id": tid, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "stage": stage, "description": description, "params": p, "sharpe_daily": np.nan}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        return tid

    def record(self, trial_id: int, sharpe: float):
        self.df.loc[self.df["trial_id"] == trial_id, "sharpe_daily"] = sharpe
        self._save()

    @property
    def n_trials(self) -> int:
        return len(self.df)

    @property
    def sr_var(self) -> float:
        s = pd.to_numeric(self.df["sharpe_daily"], errors="coerce").dropna()
        return float(s.var(ddof=1)) if len(s) >= 2 else float("nan")
