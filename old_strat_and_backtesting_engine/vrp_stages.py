"""Stage runners: build the context, run the in-sample stage (Tests 1-5, 7, 8), run the holdout once
(Test 6), evaluate the kill criteria (Step 11 / Test 9), and write the reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import old_strat_and_backtesting_engine.vrp_protocol as P
from old_strat_and_backtesting_engine.vrp_data_layer import ChainStore, align, load_market, load_options, realized_variance
from old_strat_and_backtesting_engine.vrp_metrics import TrialLog, alpha_beta, perf, sharpe_daily, trade_metrics
from old_strat_and_backtesting_engine.vrp_signals import build_signals, har_forecast


# --------------------------------------------------------------------------------------
def calendar_from_market(cfg: dict, market: dict) -> dict:
    """Trading calendar, signals and the in-sample/holdout split. Needs only SPX/VIX/VIX3M (no options),
    so the QuantConnect adapter can call it before any option data exists."""
    days_all = align(market)
    rv = realized_variance(market, days_all, cfg)
    h = cfg["har"]
    fc = har_forecast(rv, h["refit_every"], h["min_obs"], h["horizon_days"])
    vix, vix3m = market["vix"].reindex(days_all), market["vix3m"].reindex(days_all)
    sg = build_signals(vix, vix3m, fc, cfg)
    if sg["first_tradable"] is None:
        raise ValueError("Not enough history: the gate never becomes valid. Need ~1,000 trading days of "
                         "SPX/VIX history before the first trade (HAR min_obs 750 + threshold 252).")
    start = sg["first_tradable"]
    if cfg.get("trade_start"):                       # ADDED: e.g. when options data begins after the SPX history
        start = max(start, pd.Timestamp(cfg["trade_start"]))
    days = days_all[days_all >= start]
    cut = days[int(len(days) * (1 - cfg["holdout_fraction"]))]
    return {"days_all": days_all, "rv": rv, "vix": vix, "vix3m": vix3m, "signals": sg, "days": days, "cut": cut}


def build_context(cfg: dict, data_dir, outdir, trial_log_path="trial_log.csv") -> P.Context:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    market = load_market(data_dir)
    cal = calendar_from_market(cfg, market)
    days_all, days, cut, sg = cal["days_all"], cal["days"], cal["cut"], cal["signals"]
    opts, drop_log, nonpref = load_options(data_dir, cfg["option_root"])
    store = ChainStore(opts)
    px = market["spx_tr"] if market.get("spx_tr") is not None else market["spx_daily"]["close"]
    tb = market.get("tbill")
    return P.Context(
        cfg=cfg, market=market, days_all=days_all, days=days, days_is=days[days < cut], days_ho=days[days >= cut],
        chain_for=store, settle=market["settle"], signals=sg, rv=cal["rv"], vix=cal["vix"], vix3m=cal["vix3m"],
        trial_log=TrialLog(trial_log_path), drop_log=drop_log, outdir=outdir,
        spx_ret=px.pct_change(), rf_daily=(tb / 252).reindex(days_all) if tb is not None else None,
        non_preferred_root_expiries=nonpref)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict("records") if o.index.name or not isinstance(o.index, pd.RangeIndex) \
            else o.to_dict("records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _save(obj, path):
    Path(path).write_text(json.dumps(obj, default=_jsonable, indent=2))


def _cfg_hash(cfg_or_path) -> str:
    """Hash of config.yaml's bytes, or of the config dict itself (the QuantConnect route has no yaml file)."""
    if isinstance(cfg_or_path, dict):
        return hashlib.sha256(json.dumps(cfg_or_path, sort_keys=True, default=str).encode()).hexdigest()
    return hashlib.sha256(Path(cfg_or_path).read_bytes()).hexdigest()


def _pertrade(tr):
    return {k: (float(v) if np.isfinite(v) else None) for k, v in trade_metrics(tr).items()}


# --------------------------------------------------------------------------------------
def stage_insample(ctx: P.Context, config_path: str) -> dict:
    cfg, R = ctx.cfg, {}
    sig = ctx.signals["gated"]
    R["span"] = {"all_days": [str(ctx.days_all[0].date()), str(ctx.days_all[-1].date())],
                 "first_tradable": str(ctx.days[0].date()), "insample": [str(ctx.days_is[0].date()), str(ctx.days_is[-1].date())],
                 "holdout": [str(ctx.days_ho[0].date()), str(ctx.days_ho[-1].date())],
                 "n_days_is": len(ctx.days_is), "n_days_ho": len(ctx.days_ho)}

    # ---- Test 1
    R["t1_accounting"] = P.accounting_test(ctx)
    base_eq, base_tr = P._run(ctx, sig, ctx.days_is)
    R["t1_settlement"] = P.settlement_check(ctx, base_tr)
    R["t1_put_index"] = P.put_index_comparison(ctx, ctx.days_all[ctx.days_all < ctx.cut])
    R["t1_quote_sanity"] = P.quote_sanity(ctx)
    R["t1_premium"], gap = P.direct_premium_check(ctx, sig)
    R["t1_premium_with_overnight"] = P.overnight_sensitivity(ctx)

    # ---- Appendix A.1 (zero trials, in-sample only, before the full comparison)
    ung_eq, ung_tr = P._run(ctx, ctx.signals["ungated"], ctx.days_is)
    a1 = P.a1_diagnostic(ctx, gap, ung_tr)
    R["a1"] = {k: {"overall_gate_on_rate": v["overall_gate_on_rate"],
                   "gate_on_rate": v["gate_on_rate"].to_dict(), "premium": v["premium"], "per_trade": v["per_trade"]}
               for k, v in a1.items()}

    # ---- Test 2
    R["t2_lag"] = P.lag_test(ctx)
    R["t2_truncation"] = P.truncation_test(ctx)

    # ---- Test 3 (registers trials)
    table, trades, placebos, curves = P.ablation_table(ctx, ctx.days_is)
    R["t3_table"] = table
    R["t3_pertrade"] = {k: _pertrade(v) for k, v in trades.items()}
    R["t3_placebo"] = {k: v for k, v in placebos.items()}

    # ---- Test 4
    R["t4_cost_grid"] = P.cost_grid(ctx, sig, ctx.days_is)
    R["t4_f_star_insample"] = P.f_star(ctx, sig, ctx.days_is)

    # ---- Test 5 (registers every cell)
    grid = P.parameter_grid(ctx, ctx.days_is)
    R["t5_grid"] = grid
    R["t5_neighbors"] = P.neighbor_retention(ctx, grid)

    # ---- Test 7
    st, sz, cons_eq, cons_tr = P.stress_table(ctx, ctx.days_is, sig)
    R["t7_stress"], R["t7_sizing"] = st, sz
    R["t7_xsp_one_spread_risk_share"] = P.xsp_min_tradeable_risk(base_tr, cfg["start_equity"])
    R["t7_median_max_loss_pts"] = float(base_tr["max_loss_pts"].median()) if len(base_tr) else None

    # ---- Test 8 (after all trials are logged)
    R["t8_inference"] = P.inference(ctx, base_eq)
    R["gated_perf_insample"] = {k: float(v) for k, v in perf(base_eq).items()}
    R["buy_hold_and_put"] = _benchmarks(ctx, ctx.days_is)

    _save(R, ctx.outdir / "insample.json")
    _save({"config_sha256": _cfg_hash(config_path), "n_trials_at_freeze": ctx.trial_log.n_trials,
           "holdout_start": str(ctx.cut.date())}, ctx.outdir / "frozen.json")
    (ctx.outdir / "insample_report.md").write_text(render_report(R, None, None, ctx))
    return R


def _benchmarks(ctx, days):
    out = {}
    px = ctx.market["spx_tr"] if ctx.market.get("spx_tr") is not None else ctx.market["spx_daily"]["close"]
    for name, s in (("Buy and hold SPX" + ("" if ctx.market.get("spx_tr") is not None else " (price only)"), px),
                    ("PUT index", ctx.market.get("put_index"))):
        if s is None:
            continue
        lv = s.reindex(days).dropna()
        if len(lv) > 30:
            r = lv.pct_change().dropna()
            ab = alpha_beta(r, ctx.spx_ret, ctx.rf_daily)
            out[name] = {**{k: float(v) for k, v in perf(lv).items()},
                         "beta": float(ab["beta"]), "alpha_annual": float(ab["alpha_annual"]), "p_alpha": float(ab["p_alpha"])}
    return out


# --------------------------------------------------------------------------------------
def stage_holdout(ctx: P.Context, config_path: str, R_is: dict) -> tuple[dict, list, str]:
    out = ctx.outdir
    fz = json.loads((out / "frozen.json").read_text()) if (out / "frozen.json").exists() else None
    if fz is None:
        raise RuntimeError("Run the insample stage first: the design must be frozen before the holdout runs.")
    if (out / "holdout_done.json").exists():
        raise RuntimeError("The holdout has already been run. The guide says to run it once and not retune. "
                           "(Real data: do not reset this. Synthetic data: run_all.py --synthetic clears it automatically.)")
    if fz["config_sha256"] != _cfg_hash(config_path):
        raise RuntimeError("config.yaml changed after the in-sample freeze. A changed design needs a new "
                           "in-sample run (and new trial-log rows), not a holdout peek.")
    if fz["holdout_start"] != str(ctx.cut.date()):
        raise RuntimeError(f"holdout_start moved: frozen {fz['holdout_start']}, now {ctx.cut.date()}. "
                       "The data changed after the in-sample freeze.")
    cfg, days = ctx.cfg, ctx.days_ho
    R = {"span": [str(days[0].date()), str(days[-1].date())]}
    names = {"gated": "Gated spread", "ungated": "Ungated spread", "vrp_only": "VRP-only gate", "ts_only": "Term-structure-only gate"}
    rows, pertrade = [], {}
    for k, label in names.items():
        eq, tr = P._run(ctx, ctx.signals[k], days)
        rows.append({"variant": label, **P._summarize(eq, tr, ctx)})
        pertrade[k] = _pertrade(tr)
        if k == "gated":
            gated_eq, gated_tr = eq, tr
    R["table"], R["pertrade"] = pd.DataFrame(rows), pertrade
    R["gated_cost_rows"] = pd.DataFrame([{"f": f, **{kk: vv for kk, vv in P._summarize(
        *P._run(ctx, ctx.signals["gated"], days, f), ctx).items() if kk in ("CAGR", "sortino", "max_drawdown", "CVaR_5pct_daily", "worst_month", "beta", "alpha_annual", "p_alpha")}}
        for f in cfg["fill_fraction_grid"] if f != 1.0])
    R["f_star_holdout"] = P.f_star(ctx, ctx.signals["gated"], days)
    st, sz, _, cons_tr = P.stress_table(ctx, days, ctx.signals["gated"])
    R["stress"], R["sizing"] = st, sz
    R["inference"] = P.inference(ctx, gated_eq)
    R["benchmarks"] = _benchmarks(ctx, days)
    kill, verdict = evaluate_kill(R_is, R, cfg)
    _save(R, out / "holdout.json")
    _save({"verdict": verdict, "criteria": kill}, out / "decision.json")
    (out / "final_report.md").write_text(render_report(R_is, R, (kill, verdict), ctx))
    _save({"done": True}, out / "holdout_done.json")
    return R, kill, verdict


# --------------------------------------------------------------------------------------
def evaluate_kill(Ris: dict, Rho: dict | None, cfg: dict):
    k, out = cfg["kill"], []

    def add(i, name, status, detail):
        out.append({"id": i, "name": name, "status": status, "detail": detail})

    nan = lambda v: v is None or (isinstance(v, float) and not np.isfinite(v))
    g, u = Ris["t3_pertrade"]["gated"], Ris["t3_pertrade"]["ungated"]
    if not g["n_trades"]:
        add(1, "Gate beats ungated per trade", "FAIL", "gated run has no trades")
    else:
        ok = (g["avg_return_on_risk"] > u["avg_return_on_risk"] and g["loss_frequency"] <= u["loss_frequency"]
              and g["worst_trade_ror"] >= u["worst_trade_ror"])
        add(1, "Gate beats ungated per trade (f=1.0)", "PASS" if ok else "FAIL",
            f"avg ROR {g['avg_return_on_risk']:.4f} vs {u['avg_return_on_risk']:.4f}; loss freq {g['loss_frequency']:.3f} vs "
            f"{u['loss_frequency']:.3f}; worst ROR {g['worst_trade_ror']:.3f} vs {u['worst_trade_ror']:.3f}")
    p = Ris["t3_placebo"]["gated"]["p_values"]["ror"]
    add(2, "Placebo p-value (avg return on risk)", "N/A" if nan(p) else ("PASS" if p <= k["placebo_p_max"] else "FAIL"),
        f"p = {p} (max {k['placebo_p_max']}); {Ris['t3_placebo']['gated']['n_placebo']} placebo runs")
    pr = Ris["t1_premium"]
    add(3, "Direct premium check", "PASS" if (pr["mean_gap"] > 0 and pr["t_stat"] >= k["premium_tstat_min"]) else "FAIL",
        f"mean gap {pr['mean_gap']:.5f}, HAC t = {pr['t_stat']:.2f} (min {k['premium_tstat_min']})")
    fs = Ris["t4_f_star_insample"]
    add(4, "Break-even fill fraction", "FAIL" if fs < k["breakeven_fill_min"] else ("CONDITIONAL" if fs < 1.0 else "PASS"),
        f"f* in-sample = {fs:.3f}" + (f", holdout = {Rho['f_star_holdout']:.3f}" if Rho else "") +
        "; between the minimum and 1.0, proceed only if paper-trading fills beat f*")
    inf = Ris["t8_inference"]
    sig_ok = inf["p_alpha"] <= k["alpha_p_max"] and inf["boot_alpha_ci_annual"][0] > 0
    add(5, "Alpha vs SPX beta significant", "PASS" if sig_ok else "FAIL",
        f"alpha {inf['alpha_annual']:.4f}/yr, HAC p = {inf['p_alpha']:.3f}, bootstrap CI {np.round(inf['boot_alpha_ci_annual'], 4).tolist()}")
    nb = Ris["t5_neighbors"]
    if nan(nb["retention"]):
        add(6, "Parameter robustness", "FAIL" if (nb["base_avg_ror"] is None or nb["base_avg_ror"] <= 0) else "N/A",
            f"chosen-cell avg ROR {nb['base_avg_ror']}; neighbors {nb['neighbor_mean_avg_ror']}")
    else:
        add(6, "Parameter robustness", "PASS" if nb["retention"] >= k["neighbor_retention_min"] else "FAIL",
            f"neighbors keep {nb['retention']:.2f} of the chosen cell's avg ROR (min {k['neighbor_retention_min']})")
    d = inf["deflated_sharpe"]
    add(7, "Deflated Sharpe ratio", "N/A" if nan(d) else ("PASS" if d >= k["deflated_sharpe_min"] else "FAIL"),
        f"DSR = {d} with {inf['n_trials']} trials (min {k['deflated_sharpe_min']})")
    if Rho is None:
        add(8, "Holdout agrees with in-sample", "PENDING", "holdout not run yet")
    else:
        a_is, a_ho = g["avg_return_on_risk"], Rho["pertrade"]["gated"]["avg_return_on_risk"]
        if a_ho is None or a_is is None:
            add(8, "Holdout agrees with in-sample", "FAIL", "holdout gated run has no trades")
        else:
            # Interpretation: the holdout must show a positive edge AND keep a share of the in-sample edge.
            # (A holdout that agrees with a negative in-sample result is not a pass.)
            mag_ok = (a_ho > 0) and ((a_is <= 0) or (a_ho / a_is >= k["holdout_ratio_min"]))
            add(8, "Holdout agrees with in-sample", "PASS" if mag_ok else "FAIL",
                f"avg ROR in-sample {a_is:.4f} vs holdout {a_ho:.4f} ({Rho['pertrade']['gated']['n_trades']} holdout trades)")
    szs = [Ris["t7_sizing"]] + ([Rho["sizing"]] if Rho else [])
    worsts = [s["worst"] for s in szs if not nan(s["worst"])]
    if not worsts:
        add(9, "Sizing vs drawdown limit", "N/A", "no stress window falls inside the sample")
    else:
        worst = min(worsts)
        full = cfg["risk_per_trade"] * cfg["max_open"]
        L = abs(worst) / full
        max_risk = cfg["drawdown_limit"] / (L * cfg["max_open"]) if L > 0 else float("inf")
        if abs(worst) <= cfg["drawdown_limit"]:
            add(9, "Sizing vs drawdown limit", "PASS", f"worst window drawdown {worst:.3f} within limit {cfg['drawdown_limit']}")
        else:
            xsp = Ris["t7_xsp_one_spread_risk_share"]
            tradeable = xsp is not None and max_risk >= xsp
            add(9, "Sizing vs drawdown limit", "CONDITIONAL" if tradeable else "FAIL",
                f"worst {worst:.3f} breaches {cfg['drawdown_limit']}; passing risk_per_trade <= {max_risk:.4f}; one XSP spread "
                f"= {xsp:.4f} of equity. " + ("Adopt the smaller size and log the rerun as a trial." if tradeable
                                              else "The passing size is below one XSP spread."))
    fails = [o for o in out if o["status"] == "FAIL"]
    pend = [o for o in out if o["status"] in ("CONDITIONAL", "N/A", "PENDING")]
    verdict = ("REJECT" if fails else ("INCONCLUSIVE / CONDITIONAL" if pend else "PASS: proceed to paper trading"))
    return out, verdict


# --------------------------------------------------------------------------------------
def _fmt(v, d=4):
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:.{d}f}"
    return str(v)


def md_table(df: pd.DataFrame, d=4) -> str:
    if df is None or len(df) == 0:
        return "_none_\n"
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if not isinstance(df.index, pd.RangeIndex):          # keep named indexes (e.g. VIX regime labels)
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(_fmt(v, d) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def _recs(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def render_report(Ris, Rho, decision, ctx) -> str:
    L = []
    banner = ctx.market.get("_synthetic_banner")
    if banner:
        L.append(f"> **{banner}**\n")
    s = Ris["span"]
    L += ["# VRP-gated SPX put spread: results\n",
          f"Data {s['all_days'][0]} to {s['all_days'][1]}. First tradable day {s['first_tradable']}. "
          f"In-sample {s['insample'][0]} to {s['insample'][1]} ({s['n_days_is']} days); holdout {s['holdout'][0]} to {s['holdout'][1]} ({s['n_days_ho']} days).\n",
          "## Test 1: correctness and premium\n",
          f"- Accounting identity: {'passed' if Ris['t1_accounting']['passed'] else 'FAILED'} ({Ris['t1_accounting']['trades_checked']} trades)",
          f"- Settlement vs SPX close: max abs diff {Ris['t1_settlement']['max_abs_diff']}; AM-settled expiries in data: {len(Ris['t1_settlement']['am_settled_expiries_in_data'])}",
          f"- PUT index replica: {Ris['t1_put_index']}",
          f"- Direct premium: mean gap {Ris['t1_premium']['mean_gap']:.5f}, HAC t {Ris['t1_premium']['t_stat']:.2f}, share of years positive {Ris['t1_premium']['share_years_positive']:.2f}; gate on/off gap {Ris['t1_premium']['on_off']}",
          f"- Premium with overnight gap added to RV: {Ris['t1_premium_with_overnight']}\n",
          "Quote-drop fraction by year:\n", md_table(_recs(Ris["t1_quote_sanity"]), 5),
          "## Test 2: lookahead\n",
          f"- Extra one-day lag: base {Ris['t2_lag']['base']} vs lagged {Ris['t2_lag']['extra_lag']}; sign flip in avg return on risk: {Ris['t2_lag']['sign_flip_avg_ror']}",
          f"- Truncation test: all match = {Ris['t2_truncation']['all_match']} ({len(Ris['t2_truncation']['checks'])} dates)\n",
          "## Test 3: variants at f = 1.0, in-sample\n", md_table(_recs(Ris["t3_table"])),
          "Per-trade comparison:\n"]
    pt = pd.DataFrame(Ris["t3_pertrade"]).T.reset_index(names="variant")
    pt["placebo_p_ror"] = [Ris["t3_placebo"].get(v, {}).get("p_values", {}).get("ror", np.nan) for v in pt["variant"]]
    L += [md_table(pt),
          "## Test 4: cost sensitivity (gated, in-sample)\n", md_table(_recs(Ris["t4_cost_grid"])),
          f"Break-even fill fraction f*: in-sample {Ris['t4_f_star_insample']:.3f}" + (f", holdout {Rho['f_star_holdout']:.3f}" if Rho else "") + "\n",
          "## Test 5: parameter grid (f = 1.0)\n", md_table(_recs(Ris["t5_grid"])),
          f"Neighbor retention: {Ris['t5_neighbors']}\n",
          "## Test 7: stress windows (conservative marks), in-sample\n", md_table(_recs(Ris["t7_stress"])),
          f"Sizing check: {Ris['t7_sizing']}\n",
          "## Test 8: inference (gated, in-sample, f = 1.0)\n", f"```\n{json.dumps(Ris['t8_inference'], default=_jsonable, indent=1)}\n```\n",
          "## Appendix A.1: VIX regimes (in-sample)\n"]
    for k, v in Ris["a1"].items():
        L += [f"**{k}**: overall gate-on rate {v['overall_gate_on_rate']:.3f}; by regime {v['gate_on_rate']}\n",
              md_table(v["premium"] if isinstance(v["premium"], pd.DataFrame) else _recs(v["premium"])),
              md_table(v["per_trade"] if isinstance(v["per_trade"], pd.DataFrame) else _recs(v["per_trade"]))]
    L += ["Benchmarks (in-sample): " + json.dumps(Ris["buy_hold_and_put"], default=_jsonable) + "\n"]
    if Rho is not None:
        L += ["## Test 6: holdout (run once)\n", md_table(_recs(Rho["table"])), "Gated at lower fill fractions:\n",
              md_table(_recs(Rho["gated_cost_rows"])), "Holdout stress windows:\n", md_table(_recs(Rho["stress"])),
              f"Holdout sizing: {Rho['sizing']}\n"]
    if decision:
        kill, verdict = decision
        L += ["## Test 9: decision against the kill criteria\n", f"**Verdict: {verdict}**\n", md_table(pd.DataFrame(kill))]
    else:
        L += ["_Holdout not run yet; kill criterion 8 and the final decision are pending._\n"]
    return "\n".join(L)
