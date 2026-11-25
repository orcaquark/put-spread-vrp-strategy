"""Crash-probability overlay.

Target:   y_t = 1 if SPY's return over the next `horizon` sessions is <= `threshold`.
Features: momentum, trend distance, realized-vol level and ratio, drawdown from the
          252-day high, VIX level / change / z-score, VIX-minus-realized-vol spread,
          and stock-bond correlation.
Models:   regularized logistic regression, shallow gradient boosting, or their average.
Scaling:  m = clip((base_rate / p) ** alpha, m_min, 1). m is 1 when predicted crash risk
          is at or below the training base rate and falls as p rises.

Leakage control: a row is trainable only if its label window (t+1 .. t+horizon) has
completed, so the last `horizon` rows are always excluded from training (purge).
Models are refit on an expanding window by the caller, not here.

No QuantConnect dependency. scikit-learn is imported lazily; if it is missing, `fit`
returns False and `scale` returns 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FEATURE_TAIL = 330   # rows needed to compute the longest rolling feature


def _unique(s: pd.Series) -> pd.Series:
    """Drop NaNs and duplicate index labels (keep the last), sorted by date."""
    s = s.dropna()

    return s[~s.index.duplicated(keep="last")].sort_index()


def build_features(
    spx: pd.Series,
    vix: pd.Series,
    tlt: pd.Series | None = None,
) -> pd.DataFrame:
    spx = _unique(spx)
    idx = spx.index
    v = _unique(vix).reindex(idx).ffill()
    r = np.log(spx).diff()

    f = pd.DataFrame(index=idx)

    for n in (5, 21, 63, 126):
        f[f"r{n}"] = np.log(spx / spx.shift(n))

    f["dist200"] = np.log(spx / spx.rolling(200).mean())

    vol21 = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    vol126 = r.rolling(126).std() * np.sqrt(TRADING_DAYS)
    f["vol21"] = np.log(vol21)
    f["volratio"] = np.log(vol21 / vol126)
    f["dd252"] = spx / spx.rolling(252).max() - 1.0

    f["vix"] = np.log(v)
    f["vix_chg5"] = np.log(v / v.shift(5))
    f["vix_z"] = (v - v.rolling(252).mean()) / v.rolling(252).std()
    f["vrp"] = v / 100.0 - vol21

    if tlt is not None:
        t = _unique(tlt).reindex(idx).ffill()
        rt = np.log(t).diff()
        f["tlt21"] = np.log(t / t.shift(21))
        f["sbcorr"] = r.rolling(63).corr(rt)

    return f.replace([np.inf, -np.inf], np.nan)


def build_labels(spx: pd.Series, horizon: int, threshold: float) -> pd.Series:
    spx = _unique(spx)
    fwd = spx.shift(-horizon) / spx - 1.0
    y = (fwd <= threshold).astype(float)
    y[fwd.isna()] = np.nan

    return y


class CrashModel:
    def __init__(
        self,
        horizon: int = 21,
        threshold: float = -0.05,
        kind: str = "blend",          # "logit", "gbm", or "blend"
        min_obs: int = 1000,
        alpha: float = 0.5,
        m_min: float = 0.3,
    ) -> None:
        assert kind in ("logit", "gbm", "blend")
        self.horizon = horizon
        self.threshold = threshold
        self.kind = kind
        self.min_obs = min_obs
        self.alpha = alpha
        self.m_min = m_min

        self.models: list = []
        self.cols: list[str] | None = None
        self.base_rate: float | None = None
        self.n_train = 0
        self.train_auc: float | None = None

    # ------------------------------------------------------------------------------------------

    def _make_models(self) -> list:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        models = []

        if self.kind in ("logit", "blend"):
            models.append(
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.3, max_iter=1000),
                )
            )

        if self.kind in ("gbm", "blend"):
            models.append(
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=120,
                    min_samples_leaf=60,
                    l2_regularization=2.0,
                    random_state=0,
                )
            )

        return models

    def fit(
        self,
        spx: pd.Series,
        vix: pd.Series,
        tlt: pd.Series | None = None,
    ) -> bool:
        try:
            from sklearn.metrics import roc_auc_score
            models = self._make_models()
        except Exception:
            return False

        x = build_features(spx, vix, tlt)
        y = build_labels(spx, self.horizon, self.threshold)
        data = x.join(y.rename("y")).dropna()

        if len(data) < self.min_obs or data["y"].nunique() < 2:
            return False

        cols = [c for c in data.columns if c != "y"]
        xv = data[cols].to_numpy(dtype=float)
        yv = data["y"].to_numpy(dtype=float)

        for m in models:
            m.fit(xv, yv)

        self.models = models
        self.cols = cols
        self.base_rate = float(yv.mean())
        self.n_train = int(len(data))

        p = self._proba(xv)
        self.train_auc = float(roc_auc_score(yv, p))

        return True

    def _proba(self, xv: np.ndarray) -> np.ndarray:
        ps = [m.predict_proba(xv)[:, 1] for m in self.models]

        return np.mean(ps, axis=0)

    def predict(
        self,
        spx: pd.Series,
        vix: pd.Series,
        tlt: pd.Series | None = None,
    ) -> float | None:
        if not self.models or self.cols is None:
            return None

        tail = spx.dropna().tail(FEATURE_TAIL)
        x = build_features(tail, vix, tlt)

        if x.empty:
            return None

        row = x.iloc[-1].reindex(self.cols)

        if row.isna().any():
            return None

        return float(self._proba(row.to_numpy(dtype=float).reshape(1, -1))[0])

    def scale(self, p: float | None) -> float:
        if p is None or self.base_rate is None or p <= 0:
            return 1.0

        return float(np.clip((self.base_rate / p) ** self.alpha, self.m_min, 1.0))


def evaluate_predictions(
    preds: pd.Series,
    spx: pd.Series,
    horizon: int,
    threshold: float,
    base_rate: float,
) -> dict | None:
    """Out-of-sample AUC and Brier skill for logged live predictions."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    y = build_labels(spx, horizon, threshold)
    d = pd.concat([preds.rename("p"), y.rename("y")], axis=1).dropna()

    if len(d) < 100 or d["y"].nunique() < 2:
        return None

    brier = float(np.mean((d["p"] - d["y"]) ** 2))
    brier_base = float(np.mean((base_rate - d["y"]) ** 2))

    return {
        "n": int(len(d)),
        "crash_rate": float(d["y"].mean()),
        "auc": float(roc_auc_score(d["y"], d["p"])),
        "brier_skill": float(1.0 - brier / brier_base),
    }
