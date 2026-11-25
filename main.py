from AlgorithmImports import *
from datetime import date, datetime
import numpy as np
import pandas as pd

from signals import trend_signal, ewma_vol_ann, log_returns
from risk import ewma_cov, shrink_to_diag, class_budgets, risk_budget_weights
from ml_overlay import CrashModel, evaluate_predictions

# =================================================================================================
# Architecture
#
#   1. Universe       10 liquid ETFs across equities, bonds and commodities, plus BIL as cash.
#   2. Signal         Blend of time-series momentum and MA trend, tilted by cross-sectional strength.
#   3. Risk budget    Class budgets (equity / bond / commodity), inverse-vol within class.
#   4. Vol target     Portfolio scaled to TARGET_VOL using max(fast, slow) covariance.
#   5. ML overlay     Walk-forward crash-probability model scales total risk down when p is elevated.
#   6. Execution      Daily, no-trade band, sells before buys, financing charged on borrowed exposure.
#
# Evaluate in this order (do not re-tune between runs):
#   A. 2008-01-02 .. 2019-12-31   (never used to tune the previous architecture)
#   B. 2020-01-02 .. 2024-12-30   (the window used so far)
#   C. 2025-01-02 .. present      (final holdout; run once)
# Override with parameters: start=YYYY-MM-DD, end=YYYY-MM-DD.
# =================================================================================================

START_CASH = 10_000_000
HIST_START = datetime(2002, 8, 1)     # TLT inception bounds the ML training history.

# ----- Universe --------------------------------------------------------------------------------------

RISK_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD", "DBC"]
CASH_TICKER = "BIL"

ASSET_CLASS = {
    "SPY": "equity", "QQQ": "equity", "IWM": "equity",
    "EFA": "equity", "EEM": "equity", "VNQ": "equity",
    "TLT": "bond", "IEF": "bond",
    "GLD": "commodity", "DBC": "commodity",
}
CLASS_BUDGET = {"equity": 0.50, "bond": 0.25, "commodity": 0.25}

# ----- Signal ----------------------------------------------------------------------------------------

TSMOM_LOOKBACKS = (21, 63, 126, 252)
MA_WINDOWS = (50, 100, 200)
W_TSMOM = 0.5
USE_TILT = True
TILT_LOOKBACK = 126
TILT_SKIP = 21

# ----- Risk ------------------------------------------------------------------------------------------

TARGET_VOL = 0.15
MAX_GROSS = 1.75                       # Margin account allows 2x; keep headroom.
MAX_ASSET = 0.60
COV_WINDOW = 252
LAMBDA_FAST = 0.90                     # Half-life about 7 sessions.
LAMBDA_SLOW = 0.97                     # Half-life about 23 sessions.
COV_SHRINK = 0.20
VOL_LAMBDA = 0.94

# ----- ML overlay ------------------------------------------------------------------------------------

USE_ML = True
ML_KIND = "logit"                      # "logit", "gbm", "blend". Judge on the OOS log line, not train AUC.
ML_HORIZON = 21
ML_THRESHOLD = -0.05                   # Crash = SPY return <= -5% over the horizon.
ML_MIN_OBS = 1000
ML_REFIT_DAYS = 28
ML_ALPHA = 0.5
ML_M_MIN = 0.30

# ----- Execution -------------------------------------------------------------------------------------

BAND = 0.03                            # Trade a ticker only if its weight is off target by more than this.
CASH_BUFFER = 0.01
CHARGE_FINANCING = True
FINANCING_SPREAD = 0.005
TBILL_RATE_BY_YEAR = {                 # Approximate annual-average T-bill yields.
    2008: 0.014, 2009: 0.0015, 2010: 0.0014, 2011: 0.0005, 2012: 0.0009,
    2013: 0.0006, 2014: 0.0003, 2015: 0.0005, 2016: 0.0032, 2017: 0.0093,
    2018: 0.0194, 2019: 0.0206, 2020: 0.0037, 2021: 0.0004, 2022: 0.020,
    2023: 0.051, 2024: 0.050, 2025: 0.042, 2026: 0.037,
}
TBILL_RATE_DEFAULT = 0.03


class MultiAssetTrendRisk(QCAlgorithm):

    def initialize(self):
        start = date.fromisoformat(self.get_parameter("start", "2025-01-02"))
        end = date.fromisoformat(self.get_parameter("end", "2026-11-20"))
        self.end_day = end

        self.set_start_date(start.year, start.month, start.day)
        self.set_end_date(end.year, end.month, end.day)
        self.set_cash(START_CASH)

        self.tickers = RISK_TICKERS + [CASH_TICKER]
        self.sym = {}

        for t in self.tickers:
            eq = self.add_equity(t, Resolution.MINUTE)
            eq.set_leverage(2.0)
            self.sym[t] = eq.symbol

        self.vix = self.add_index("VIX", Resolution.DAILY).symbol
        self.set_benchmark("SPY")

        self.px = None
        self.crash = CrashModel(
            horizon=ML_HORIZON,
            threshold=ML_THRESHOLD,
            kind=ML_KIND,
            min_obs=ML_MIN_OBS,
            alpha=ML_ALPHA,
            m_min=ML_M_MIN,
        )
        self.last_fit = None
        self.fit_ok = False
        self.pred_log = {}

        self.gross_hist = []
        self.mult_hist = []
        self.traded_value = 0.0
        self.financing_paid = 0.0
        self.first_equity = None
        self.last_log_month = None

        spy = self.sym["SPY"]
        self.schedule.on(
            self.date_rules.every_day(spy),
            self.time_rules.at(15, 50),
            self.rebalance,
        )
        self.schedule.on(
            self.date_rules.every_day(spy),
            self.time_rules.at(15, 58),
            self.charge_financing,
        )

    # ================================================================================================
    # Data
    # ================================================================================================

    @staticmethod
    def _name(x):
        return getattr(x, "value", str(x))

    def _closes(self, hist):
        if hist is None or len(hist) == 0:
            return pd.DataFrame()

        c = hist["close"].unstack(level=0)
        c.columns = [self._name(x) for x in c.columns]
        c.index = pd.DatetimeIndex(pd.to_datetime(c.index).normalize())

        # Equities and the VIX index can carry different bar timestamps for the same session.
        # After normalizing to dates, merge those rows (last non-null value per column).
        return c.groupby(level=0).last().sort_index()

    def _update_prices(self, today):
        syms = list(self.sym.values()) + [self.vix]

        if self.px is None:
            hist = self.history(syms, HIST_START, self.time, Resolution.DAILY)
            self.px = self._closes(hist)
        else:
            new = self._closes(self.history(syms, 10, Resolution.DAILY))
            self.px = pd.concat([self.px, new]).groupby(level=0).last().sort_index()

        # Completed sessions only.
        self.px = self.px[self.px.index < pd.Timestamp(today)]

    # ================================================================================================
    # Portfolio construction
    # ================================================================================================

    def _target_weights(self, px):
        risk_px = px[[t for t in RISK_TICKERS if t in px.columns]].ffill(limit=5)

        if len(risk_px) < 300:
            return None, None

        rets = log_returns(risk_px)
        win = rets.tail(COV_WINDOW)
        valid = list(win.columns[win.notna().all()])

        if len(valid) < 3:
            return None, None

        sig = trend_signal(
            risk_px[valid],
            w_tsmom=W_TSMOM,
            lookbacks=TSMOM_LOOKBACKS,
            ma_windows=MA_WINDOWS,
            use_tilt=USE_TILT,
            tilt_lookback=TILT_LOOKBACK,
            tilt_skip=TILT_SKIP,
        )
        vol = ewma_vol_ann(rets[valid], VOL_LAMBDA)

        cov_fast = ewma_cov(win[valid], LAMBDA_FAST)
        cov_slow = ewma_cov(win[valid], LAMBDA_SLOW)

        if cov_fast is None or cov_slow is None:
            return None, None

        cov_fast = shrink_to_diag(cov_fast, COV_SHRINK)
        cov_slow = shrink_to_diag(cov_slow, COV_SHRINK)

        budgets = class_budgets(valid, ASSET_CLASS, CLASS_BUDGET)

        w, info = risk_budget_weights(
            signal=sig,
            vol=vol,
            budgets=budgets,
            cov_fast=cov_fast,
            cov_slow=cov_slow,
            target_vol=TARGET_VOL,
            max_gross=MAX_GROSS,
            max_asset=MAX_ASSET,
        )

        return w.reindex(RISK_TICKERS).fillna(0.0), info

    def _ml_multiplier(self, px, today):
        if not USE_ML or "SPY" not in px.columns or "VIX" not in px.columns:
            return 1.0, None

        spx = px["SPY"].dropna()
        vix = px["VIX"].ffill()
        tlt = px["TLT"] if "TLT" in px.columns else None

        due = self.last_fit is None or (today - self.last_fit).days >= ML_REFIT_DAYS

        if due:
            self.fit_ok = self.crash.fit(spx, vix, tlt)
            self.last_fit = today

            if self.fit_ok:
                self.log(
                    f"ml refit {today} n={self.crash.n_train} "
                    f"base_rate={self.crash.base_rate:.3f} "
                    f"in_sample_auc={self.crash.train_auc:.3f}"
                )

        if not self.fit_ok:
            return 1.0, None

        p = self.crash.predict(spx, vix, tlt)

        if p is not None:
            self.pred_log[pd.Timestamp(today)] = p

        return self.crash.scale(p), p

    # ================================================================================================
    # Rebalance
    # ================================================================================================

    def rebalance(self):
        today = self.time.date()

        if today > self.end_day:
            self.quit("Date is past the configured backtest end.")
            return

        self._update_prices(today)

        w, info = self._target_weights(self.px)

        if w is None:
            return

        m, p = self._ml_multiplier(self.px, today)
        w = w * m

        equity = float(self.portfolio.total_portfolio_value)

        if equity <= 0:
            return

        if self.first_equity is None:
            self.first_equity = equity

        gross = float(w.sum())
        targets = {t: float(w[t]) for t in RISK_TICKERS}
        targets[CASH_TICKER] = max(0.0, 1.0 - gross - CASH_BUFFER)

        moves = []

        for t, target in targets.items():
            current = self.portfolio[self.sym[t]].holdings_value / equity

            if abs(target - current) > BAND or (target == 0.0 and current > 0.005):
                moves.append((target - current, t, target))

        # Reductions first so freed margin is available for purchases.
        for _, t, target in sorted(moves, key=lambda x: x[0]):
            self.set_holdings(self.sym[t], target)

        self.gross_hist.append(gross)
        self.mult_hist.append(m)

        self.plot("Portfolio Gross", "gross", gross)
        self.plot("Portfolio Gross", "cash_bil", targets[CASH_TICKER])
        self.plot("Model Signals", "ml_multiplier", m)
        self.plot("Model Signals", "ml_crash_prob", p if p is not None else 0.0)
        self.plot("Model Signals", "vol_scale_k", info["k"])

        month = (today.year, today.month)

        if month != self.last_log_month:
            self.last_log_month = month
            top = ", ".join(
                f"{t}={targets[t]:.2f}"
                for t in sorted(RISK_TICKERS, key=lambda x: -targets[x])
                if targets[t] > 0.005
            )
            self.log(
                f"{today} gross={gross:.2f} ml_m={m:.2f} "
                f"p={'na' if p is None else f'{p:.3f}'} k={info['k']:.2f} | {top}"
            )

    # ================================================================================================
    # Financing and diagnostics
    # ================================================================================================

    def charge_financing(self):
        """Debit cash for borrowed long exposure; LEAN charges no margin interest by default."""
        if not CHARGE_FINANCING:
            return

        equity = float(self.portfolio.total_portfolio_value)
        longs = sum(
            float(self.portfolio[s].holdings_value) for s in self.sym.values()
        )
        borrowed = max(0.0, longs - equity)

        if borrowed <= 0.0:
            return

        rate = TBILL_RATE_BY_YEAR.get(self.time.year, TBILL_RATE_DEFAULT) + FINANCING_SPREAD
        cost = borrowed * rate / 252.0
        self.portfolio.cash_book["USD"].add_amount(-cost)
        self.financing_paid += cost

    def on_order_event(self, order_event):
        if order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            self.log(
                f"order {order_event.status} {order_event.symbol} "
                f"qty={order_event.quantity} {order_event.message}"
            )
            return

        if order_event.status == OrderStatus.FILLED:
            self.traded_value += abs(
                float(order_event.fill_price) * float(order_event.fill_quantity)
            )

    def on_end_of_algorithm(self):
        years = max((self.end_day - self.start_date.date()).days / 365.25, 1e-9)
        avg_equity = float(self.portfolio.total_portfolio_value)

        self.log(
            "config: "
            f"target_vol={TARGET_VOL} max_gross={MAX_GROSS} class_budget={CLASS_BUDGET} "
            f"ml={USE_ML}/{ML_KIND} alpha={ML_ALPHA} m_min={ML_M_MIN} band={BAND}"
        )

        if self.gross_hist:
            self.log(
                f"mean_gross={np.mean(self.gross_hist):.2f} "
                f"max_gross={np.max(self.gross_hist):.2f} "
                f"mean_ml_multiplier={np.mean(self.mult_hist):.3f} "
                f"turnover_per_year={self.traded_value / avg_equity / years:.2f}"
            )

        self.log(f"financing_paid_usd={self.financing_paid:,.0f}")

        if self.pred_log and self.crash.base_rate is not None and self.px is not None:
            preds = pd.Series(self.pred_log).sort_index()
            report = evaluate_predictions(
                preds,
                self.px["SPY"].dropna(),
                ML_HORIZON,
                ML_THRESHOLD,
                self.crash.base_rate,
            )
            self.log(f"ml out-of-sample: {report}")