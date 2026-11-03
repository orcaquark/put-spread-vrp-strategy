"""Testing protocol (guide Section 5). Each function takes a Context and returns plain data."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from vrp_engine import run_backtest
from vrp_execution import fill_price
from vrp_metrics import (alpha_beta, bootstrap_alpha_ci, bootstrap_mean_ci, breakeven_fill_fraction,
                      deflated_sharpe, perf, sharpe_daily, trade_metrics)
from vrp_signals import build_signals, forward_mean, har_forecast


@dataclass
class Context:
    cfg: dict
    market: dict
    days_all: pd.DatetimeIndex
    days: pd.DatetimeIndex           # tradable days (from the first date the gate can fire)
    days_is: pd.DatetimeIndex
    days_ho: pd.DatetimeIndex
    chain_for: object
    settle: pd.Series
    signals: dict
    rv: pd.Series
    vix: pd.Series
    vix3m: pd.Series
    trial_log: object
    drop_log: pd.DataFrame
    outdir: Path
    spx_ret: pd.Series
    rf_daily: pd.Series | None
    non_preferred_root_expiries: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)

    @property
    def cut(self):
        return self.days_ho[0]


def _run(ctx, sig, days, f=1.0, cfg=None, mark="mid"):
    return run_backtest(days, ctx.chain_for, ctx.settle, sig, cfg or ctx.cfg, f, ctx.cache, mark)


def variant_params(cfg, gate):
    return {"gate": gate, "short_delta": cfg["short_delta"], "long_delta": cfg["long_delta"],
            "dte": cfg["target_dte"]}


def _summarize(eq, tr, ctx):
    r = eq.pct_change().dropna()
    return {**perf(eq), **trade_metrics(tr), **{k: v for k, v in alpha_beta(
        r, ctx.spx_ret, ctx.rf_daily).items() if k in ("alpha_annual", "p_alpha", "beta")}}


# ======================================================================================
# Test 1: correctness and premium checks
# ======================================================================================
def premium_gap(vix: pd.Series, rv: pd.Series, h: int = 22) -> pd.Series:
    """Implied minus subsequently realized variance, annualized. Positive means options overpaid."""
    implied = (vix / 100) ** 2
    realized_next = 252 * forward_mean(rv, h)
    return (implied - realized_next).dropna()


def direct_premium_check(ctx, signal):
    """Premium check on in-sample data only (rv is cut at the holdout start so no holdout variance leaks in)."""
    h = ctx.cfg["har"]["horizon_days"]
    pre = ctx.days_all[ctx.days_all < ctx.cut]
    gap = premium_gap(ctx.vix.loc[pre], ctx.rv.loc[pre], h)
    fit = sm.OLS(gap.values, np.ones(len(gap))).fit(cov_type="HAC", cov_kwds={"maxlags": 22})
    out = {"mean_gap": float(fit.params[0]), "t_stat": float(fit.tvalues[0]), "n": int(len(gap)),
           "share_years_positive": float((gap.groupby(gap.index.year).mean() > 0).mean()),
           "by_year": gap.groupby(gap.index.year).mean().round(5).to_dict(),
           "on_off": {str(k): float(v) for k, v in gap.groupby(
               signal.reindex(gap.index, fill_value=False)).mean().items()}}
    return out, gap


def overnight_sensitivity(ctx):
    """Same premium check with the overnight gap added to RV (see IMPLEMENTATION_NOTES.md, issue 4)."""
    from vrp_data_layer import realized_variance
    if ctx.cfg["rv"]["estimator"] != "5min":
        return None
    rv_o = realized_variance(ctx.market, ctx.days_all, ctx.cfg, include_overnight=True)
    pre = ctx.days_all[ctx.days_all < ctx.cut]
    gap = premium_gap(ctx.vix.loc[pre], rv_o.loc[pre], ctx.cfg["har"]["horizon_days"])
    fit = sm.OLS(gap.values, np.ones(len(gap))).fit(cov_type="HAC", cov_kwds={"maxlags": 22})
    return {"mean_gap": float(fit.params[0]), "t_stat": float(fit.tvalues[0])}


def check_accounting(equity: pd.Series, trades: pd.DataFrame, cfg: dict) -> None:
    pnl_usd = (trades["pnl_pts"] * 100 * trades["contracts"]).sum()
    expected = cfg["start_equity"] + pnl_usd - trades["fees"].sum()
    assert abs(equity.iloc[-1] - expected) <= 1e-6 * max(1.0, abs(expected)), (equity.iloc[-1], expected)
    width = trades["k_short"] - trades["k_long"]
    assert (trades["credit_pts"] < width).all()                          # credit can't exceed width
    assert (trades["pnl_pts"] >= -trades["max_loss_pts"] - 1e-9).all()   # loss never exceeds max loss
    assert (trades["pnl_pts"] <= trades["credit_pts"] + 1e-9).all()      # profit never exceeds the credit


def accounting_test(ctx):
    """Run on a window that ends after the last spread expires: extend ~35 calendar days with the signal off.
    The extension days only settle in-sample spreads; no performance number comes from them."""
    last = ctx.days_is[-1]
    ext = ctx.days_all[(ctx.days_all >= ctx.days_is[0]) & (ctx.days_all <= last + pd.Timedelta(days=35))]
    sig = ctx.signals["gated"].reindex(ext, fill_value=False).copy()
    sig.loc[sig.index > last] = False
    eq, tr = _run(ctx, sig, ext)
    check_accounting(eq, tr, ctx.cfg)
    return {"trades_checked": int(len(tr)), "passed": True}


def settlement_check(ctx, trades):
    close = ctx.market["spx_daily"]["close"]
    rows = []
    for e in pd.DatetimeIndex(trades["expiry"].unique()):
        if e in ctx.settle.index and e in close.index:
            rows.append((e, float(ctx.settle.loc[e]), float(close.loc[e])))
    df = pd.DataFrame(rows, columns=["expiry", "settle", "spx_close"])
    df["diff"] = (df["settle"] - df["spx_close"]).abs()
    bad = df[df["diff"] > 0.05]
    return {"n_expiries": int(len(df)), "max_abs_diff": float(df["diff"].max()) if len(df) else None,
            "expiries_not_equal_to_close": [str(x.date()) for x in bad["expiry"]][:20],
            "am_settled_expiries_in_data": ctx.non_preferred_root_expiries[:20],
            "note": "PM-settled (SPXW) settlement should equal the SPX close; AM-settled expiries should NOT."}


def put_write_replica(ctx, days, f=0.0):
    """Sell one ATM put at each third-Friday roll, expiring at the next third-Friday, collateralized at K.
    Fill at mid by default (f=0)."""
    exps = [e for e in ctx.settle.index if 15 <= e.day <= 21 and e.weekday() == 4]
    exps = [e for e in exps if e >= days[0]]
    rows = []
    close = ctx.market["spx_daily"]["close"]
    for r0, r1 in zip(exps[:-1], exps[1:]):
        if r0 not in days or r0 not in close.index:
            continue
        ch = ctx.chain_for(r0)
        puts = ch[(ch["expiry"] == r1) & (ch["cp"] == "P") & (ch["ask"] > 0)]
        if puts.empty or r1 not in ctx.settle.index:
            continue
        row = puts.iloc[(puts["strike"] - close.loc[r0]).abs().argmin()]
        K, prem = float(row["strike"]), fill_price(row, "sell", f)
        pnl = prem - max(K - ctx.settle.loc[r1], 0.0)
        rf = 0.0
        if ctx.market.get("tbill") is not None and r0 in ctx.market["tbill"].index:
            rf = ctx.market["tbill"].loc[r0] * (r1 - r0).days / 365
        rows.append((r0, r1, pnl / K + rf))
    return pd.DataFrame(rows, columns=["roll", "expiry", "ret"])


def put_index_comparison(ctx, days):
    rep = put_write_replica(ctx, days)
    pi = ctx.market.get("put_index")
    if pi is None:
        return {"status": "skipped: put_index.parquet not provided", "n_rolls": int(len(rep))}
    ref = []
    for _, r in rep.iterrows():
        if r["roll"] in pi.index and r["expiry"] in pi.index:
            ref.append(pi.loc[r["expiry"]] / pi.loc[r["roll"]] - 1)
        else:
            ref.append(np.nan)
    rep["put_index_ret"] = ref
    rep = rep.dropna()
    corr = float(rep["ret"].corr(rep["put_index_ret"])) if len(rep) > 2 else np.nan
    return {"status": "ok", "n_rolls": int(len(rep)), "correlation": corr, "target": 0.95,
            "mean_replica": float(rep["ret"].mean()), "mean_put_index": float(rep["put_index_ret"].mean())}


def quote_sanity(ctx):
    dl = ctx.drop_log
    g = dl.groupby("year").agg(n_dropped=("n_dropped", "sum"), n_rows=("n_rows", "first"))
    g["fraction_dropped"] = g["n_dropped"] / g["n_rows"]
    prev = g["fraction_dropped"].shift(1)
    g["flag"] = (g["fraction_dropped"] > 0.01) & (g["fraction_dropped"] > 2 * prev.fillna(0))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(g.index.astype(int), g["fraction_dropped"])
    ax.set_title("Fraction of option rows dropped by year (Step 3)")
    fig.tight_layout()
    fig.savefig(ctx.outdir / "drop_fraction_by_year.png", dpi=110)
    plt.close(fig)
    return g.reset_index()


# ======================================================================================
# Test 2: lookahead checks
# ======================================================================================
def lag_test(ctx, sig=None):
    sig = ctx.signals["gated"] if sig is None else sig
    base_eq, base_tr = _run(ctx, sig, ctx.days_is)
    lag_eq, lag_tr = _run(ctx, sig.shift(1, fill_value=False), ctx.days_is)
    b, l = _summarize(base_eq, base_tr, ctx), _summarize(lag_eq, lag_tr, ctx)
    keys = ["n_trades", "avg_return_on_risk", "loss_frequency", "CAGR", "sortino", "max_drawdown"]
    return {"base": {k: float(b[k]) for k in keys}, "extra_lag": {k: float(l[k]) for k in keys},
            "sign_flip_avg_ror": bool(np.sign(b["avg_return_on_risk"]) != np.sign(l["avg_return_on_risk"]))}


def truncation_test(ctx):
    """Signal computed from data truncated at t must equal the full-sample signal at t, exactly."""
    t_cfg = ctx.cfg["truncation_test"]
    rng = np.random.default_rng(t_cfg["seed"])
    cand = ctx.days_is[ctx.days_is.get_loc(ctx.days_is[0]):]
    picks = sorted(rng.choice(len(cand), size=min(t_cfg["n_dates"], len(cand)), replace=False))
    h = ctx.cfg["har"]
    out = []
    for i in picks:
        t = cand[i]
        sub = ctx.days_all[ctx.days_all <= t]
        fc = har_forecast(ctx.rv.loc[sub], h["refit_every"], h["min_obs"], h["horizon_days"])
        sg = build_signals(ctx.vix.loc[sub], ctx.vix3m.loc[sub], fc, ctx.cfg)
        full = ctx.signals
        d_vrp = abs(sg["vrp"].iloc[-1] - full["vrp"].loc[t])
        d_thr = abs(sg["threshold"].iloc[-1] - full["threshold"].loc[t])
        out.append({"date": str(t.date()), "signal_match": bool(sg["gated"].iloc[-1] == full["gated"].loc[t]),
                    "max_abs_diff_vrp": float(np.nan_to_num(d_vrp)), "max_abs_diff_threshold": float(np.nan_to_num(d_thr))})
    return {"checks": out, "all_match": all(o["signal_match"] and o["max_abs_diff_vrp"] < 1e-12
                                            and o["max_abs_diff_threshold"] < 1e-12 for o in out)}


# ======================================================================================
# Test 3: does the gate help?
# ======================================================================================
def placebo_test(ctx, sig, days, label):
    pc = ctx.cfg["placebo"]
    rng = np.random.default_rng(pc["seed"])
    base = sig.reindex(days, fill_value=False).to_numpy()
    eq, tr = _run(ctx, sig, days)
    if len(tr) == 0:
        return {"label": label, "p_values": {"cvar": np.nan, "ror": np.nan}, "n_placebo": 0, "real": {}}
    real = {"cvar": perf(eq)["CVaR_5pct_daily"], "ror": trade_metrics(tr)["avg_return_on_risk"]}
    scores = {"cvar": [], "ror": []}
    lo, hi = pc["min_shift"], len(base) - pc["min_shift"]
    for _ in range(pc["reps"]):
        k = int(rng.integers(lo, hi))
        fake = pd.Series(np.roll(base, k), index=days)       # preserves hit rate and persistence
        e, t = _run(ctx, fake, days)
        if len(t) == 0:
            continue                                         # drop placebo runs with no trades
        scores["cvar"].append(perf(e)["CVaR_5pct_daily"])
        scores["ror"].append(trade_metrics(t)["avg_return_on_risk"])
    p = {m: float((np.array(v) >= real[m]).mean()) if v else np.nan for m, v in scores.items()}
    return {"label": label, "p_values": p, "n_placebo": len(scores["ror"]), "real": real}


def ablation_table(ctx, days, f=1.0, with_placebo=True):
    rows, trades, placebos, curves = [], {}, {}, {}
    names = {"gated": "Gated spread", "ungated": "Ungated spread", "vrp_only": "VRP-only gate",
             "ts_only": "Term-structure-only gate"}
    for key, label in names.items():
        tid = ctx.trial_log.register(f"{key}_spread", variant_params(ctx.cfg, key), "insample")
        eq, tr = _run(ctx, ctx.signals[key], days, f)
        rows.append({"variant": label, **_summarize(eq, tr, ctx)})
        trades[key], curves[key] = tr, eq
        if days is ctx.days_is:
            ctx.trial_log.record(tid, sharpe_daily(eq.pct_change().dropna()))
        if with_placebo and key != "ungated":
            placebos[key] = placebo_test(ctx, ctx.signals[key], days, label)
    return pd.DataFrame(rows), trades, placebos, curves


# ======================================================================================
# Test 4: cost sensitivity
# ======================================================================================
def cost_grid(ctx, sig, days):
    cfg = ctx.cfg
    rows = []
    for f in cfg["fill_fraction_grid"]:
        for m in cfg["cost_grid"]["fee_multipliers"]:
            c = {**cfg, "fee_per_contract": cfg["fee_per_contract"] * m}
            eq, tr = _run(ctx, sig, days, f, c)
            rows.append({"f": f, "fee_x": m, "net_pnl_usd": float(eq.iloc[-1] - cfg["start_equity"]),
                         "CAGR": perf(eq)["CAGR"], "sortino": perf(eq)["sortino"],
                         "n_trades": len(tr), "avg_return_on_risk": trade_metrics(tr)["avg_return_on_risk"]})
    return pd.DataFrame(rows)


def f_star(ctx, sig, days):
    return float(breakeven_fill_fraction(
        lambda f: _run(ctx, sig, days, f)[0].iloc[-1] - ctx.cfg["start_equity"]))


# ======================================================================================
# Test 5: parameter robustness
# ======================================================================================
def parameter_grid(ctx, days):
    g = ctx.cfg["grid"]
    rows = []
    for dte in g["dte"]:
        for sd in g["short_delta"]:
            for ld in g["long_delta"]:
                if ld >= sd:
                    continue
                c = {**ctx.cfg, "short_delta": sd, "long_delta": ld, "target_dte": dte}
                tid = ctx.trial_log.register("gated_spread", variant_params(c, "gated"), "insample")
                eq, tr = _run(ctx, ctx.signals["gated"], days, 1.0, c)
                p, t = perf(eq), trade_metrics(tr)
                ctx.trial_log.record(tid, sharpe_daily(eq.pct_change().dropna()))
                rows.append({"dte": dte, "short_delta": sd, "long_delta": ld, "sortino": p["sortino"],
                             "CVaR_5pct_daily": p["CVaR_5pct_daily"], "avg_return_on_risk": t["avg_return_on_risk"],
                             "n_trades": t["n_trades"]})
    df = pd.DataFrame(rows)
    _heatmaps(ctx, df)
    return df


def _heatmaps(ctx, df):
    dtes = sorted(df["dte"].unique())
    for metric in ("sortino", "CVaR_5pct_daily"):
        fig, axes = plt.subplots(1, len(dtes), figsize=(4.2 * len(dtes), 3.4), squeeze=False)
        for ax, dte in zip(axes[0], dtes):
            pv = df[df["dte"] == dte].pivot(index="short_delta", columns="long_delta", values=metric)
            im = ax.imshow(pv.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pv.columns)), [f"{c:.2f}" for c in pv.columns])
            ax.set_yticks(range(len(pv.index)), [f"{c:.2f}" for c in pv.index])
            ax.set_title(f"{metric} | DTE {dte}")
            ax.set_xlabel("long delta")
            ax.set_ylabel("short delta")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(ctx.outdir / f"heatmap_{metric}.png", dpi=110)
        plt.close(fig)


def neighbor_retention(ctx, grid_df):
    """Edge = average return on risk. Neighbors differ from the chosen cell by one grid step in one parameter."""
    g, cfg = ctx.cfg["grid"], ctx.cfg
    def idx(lst, v):
        return min(range(len(lst)), key=lambda i: abs(lst[i] - v))
    i_sd, i_ld, i_dte = idx(g["short_delta"], cfg["short_delta"]), idx(g["long_delta"], cfg["long_delta"]), idx(g["dte"], cfg["target_dte"])
    def cell(sd, ld, dte):
        r = grid_df[(grid_df["short_delta"] == sd) & (grid_df["long_delta"] == ld) & (grid_df["dte"] == dte)]
        return float(r["avg_return_on_risk"].iloc[0]) if len(r) else np.nan
    base = cell(g["short_delta"][i_sd], g["long_delta"][i_ld], g["dte"][i_dte])
    nbrs = []
    for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        a, b, c = i_sd + di, i_ld + dj, i_dte + dk
        if 0 <= a < len(g["short_delta"]) and 0 <= b < len(g["long_delta"]) and 0 <= c < len(g["dte"]):
            v = cell(g["short_delta"][a], g["long_delta"][b], g["dte"][c])
            if np.isfinite(v):
                nbrs.append(v)
    mean_n = float(np.mean(nbrs)) if nbrs else np.nan
    ret = mean_n / base if base and base > 0 and np.isfinite(mean_n) else np.nan
    return {"base_avg_ror": base, "neighbor_mean_avg_ror": mean_n, "n_neighbors": len(nbrs), "retention": ret}


# ======================================================================================
# Test 7: stress windows and sizing check
# ======================================================================================
WINDOWS = {
    "gfc_2008":   ("2008-09-01", "2008-12-31"),
    "aug_2015":   ("2015-08-17", "2015-08-25"),
    "feb_2018":   ("2018-02-02", "2018-02-09"),
    "q4_2018":    ("2018-10-01", "2018-12-31"),
    "covid_2020": ("2020-02-20", "2020-03-23"),
    "bear_2022":  ("2022-01-01", "2022-12-31"),
}


def sizing_check(equity: pd.Series, cfg: dict, windows: dict = WINDOWS) -> dict:
    full_ladder = cfg["risk_per_trade"] * cfg["max_open"]
    dd = {}
    for name, (start, end) in windows.items():
        w = equity.loc[start:end]
        if len(w) > 1:
            dd[name] = (w / w.cummax() - 1).min()
    if not dd:
        return {"window_drawdowns": {}, "worst": np.nan, "L": np.nan, "max_risk_per_trade": np.nan, "passes": None,
                "note": "no stress window falls inside this sample"}
    worst = min(dd.values())                              # most negative drawdown
    L = abs(worst) / full_ladder
    max_risk = cfg["drawdown_limit"] / (L * cfg["max_open"]) if L > 0 else float("inf")
    return {"window_drawdowns": dd, "worst": worst, "L": L,
            "max_risk_per_trade": max_risk, "passes": abs(worst) <= cfg["drawdown_limit"]}


def stress_table(ctx, days, sig):
    """Most conservative equity curve (short leg marked at ask, long at bid). Per window: worst drawdown,
    spreads that settled at maximum loss among those open during the window, gate state going in."""
    eq, tr = _run(ctx, sig, days, 1.0, None, "conservative")
    sz = sizing_check(eq, ctx.cfg)
    rows = []
    for name, (s, e) in WINDOWS.items():
        w = eq.loc[s:e]
        if len(w) < 2:
            continue
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        opened = tr[(tr["open"] <= e_ts) & (tr["expiry"] >= s_ts)]
        at_max = int((opened["pnl_pts"] <= -opened["max_loss_pts"] + 1e-6).sum())
        prior = sig.loc[:s_ts]
        rows.append({"window": name, "worst_drawdown": float((w / w.cummax() - 1).min()),
                     "spreads_open_in_window": int(len(opened)), "spreads_at_max_loss": at_max,
                     "gate_on_going_in": bool(prior.iloc[-1]) if len(prior) else None})
    return pd.DataFrame(rows), sz, eq, tr


def xsp_min_tradeable_risk(trades: pd.DataFrame, equity: float) -> float:
    """Risk of ONE XSP spread (1/10 SPX notional) as a share of equity, from the median trade's max loss."""
    if len(trades) == 0:
        return float("nan")
    return float(trades["max_loss_pts"].median() * 10 / equity)


# ======================================================================================
# Test 8: statistical inference
# ======================================================================================
def inference(ctx, eq):
    cfg, bs = ctx.cfg, ctx.cfg["bootstrap"]
    r = eq.pct_change().dropna()
    ab = alpha_beta(r, ctx.spx_ret, ctx.rf_daily)
    lo_m, hi_m = bootstrap_mean_ci(r, bs["block"], bs["reps"], bs["seed"])
    lo_a, hi_a = bootstrap_alpha_ci(r, ctx.spx_ret, ctx.rf_daily, bs["block"], bs["reps"], bs["seed"])
    sr = sharpe_daily(r)
    dsr = deflated_sharpe(sr, len(r), float(r.skew()), float(r.kurt() + 3.0), ctx.trial_log.n_trials,
                          ctx.trial_log.sr_var)
    return {**ab, "boot_mean_ci_annual": [lo_m, hi_m], "boot_alpha_ci_annual": [lo_a, hi_a],
            "sharpe_daily": sr, "sharpe_annualized": sr * np.sqrt(252), "n_trials": ctx.trial_log.n_trials,
            "sr_var_across_trials": ctx.trial_log.sr_var, "deflated_sharpe": dsr}


# ======================================================================================
# Appendix A.1: VIX-regime diagnostic (zero trials; in-sample only)
# ======================================================================================
def vix_regime(vix: pd.Series, min_periods: int = 252) -> pd.Series:
    lo = vix.expanding(min_periods=min_periods).quantile(1 / 3).shift(1)
    hi = vix.expanding(min_periods=min_periods).quantile(2 / 3).shift(1)
    regime = pd.Series("mid", index=vix.index, dtype=object)
    regime[vix <= lo] = "low"
    regime[vix > hi] = "high"
    regime[lo.isna()] = np.nan                 # not enough history yet
    return regime


def high_vol_regime(vix: pd.Series, q: float = 0.8, min_periods: int = 252) -> pd.Series:
    thr = vix.expanding(min_periods=min_periods).quantile(q).shift(1)
    regime = pd.Series(np.where(vix > thr, "top20", "rest"), index=vix.index, dtype=object)
    regime[thr.isna()] = np.nan
    return regime


def regime_table(gap, implied, reg, order=("low", "mid", "high")):
    df = pd.DataFrame({"gap": gap, "rel_gap": gap / implied, "regime": reg}).dropna()
    g = df.groupby("regime")
    return pd.DataFrame({
        "days": g.size(), "mean_gap": g["gap"].mean(), "std_gap": g["gap"].std(),
        "gap_per_risk": g["gap"].mean() / g["gap"].std(), "mean_rel_gap": g["rel_gap"].mean(),
    }).reindex(list(order))


def a1_diagnostic(ctx, gap, trades_ungated):
    cut = ctx.cut
    vix_in = ctx.vix[ctx.vix.index < cut]
    implied_in = (vix_in / 100) ** 2
    gap_in = gap[gap.index < cut]
    signal_in = ctx.signals["gated"][ctx.signals["gated"].index < cut]
    trades_in = trades_ungated[trades_ungated["open"] < cut]
    out = {}
    for label, make, order in (("terciles", vix_regime, ("low", "mid", "high")),
                               ("top20", high_vol_regime, ("rest", "top20"))):
        reg = make(vix_in).shift(1)
        on_rate = signal_in.groupby(reg.reindex(signal_in.index)).mean()
        t = trades_in.copy()
        t["regime"] = reg.reindex(t["open"]).values
        t["ror"] = t["pnl_pts"] / t["max_loss_pts"]
        per_trade = t.groupby("regime").agg(n=("ror", "size"), avg_ror=("ror", "mean"),
                                            loss_freq=("pnl_pts", lambda s: (s < 0).mean()),
                                            worst_ror=("ror", "min"))
        out[label] = {"overall_gate_on_rate": float(signal_in.mean()), "gate_on_rate": on_rate,
                      "premium": regime_table(gap_in, implied_in, reg, order), "per_trade": per_trade}
    return out
