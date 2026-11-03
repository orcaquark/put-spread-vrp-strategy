"""QuantConnect Research adapter (route 1): turn a data source into the processed-data schema.

Why plan-then-fetch: QuantConnect's daily option universe carries prices, IV and Greeks for the whole chain
but no documented bid/ask; bid/ask exists only per contract (QuoteBar history). Fetching quotes for every
contract is impractical, so this adapter
  1. plans the exact legs the strategy would pick on every possible entry date, using the universe's delta,
  2. fetches quote history only for those legs (resumable, cached on disk),
  3. writes options_<YYYY>.parquet containing only those contracts, plus spx_daily/vix/vix3m/settlement,
and the unmodified pipeline (stages.build_context / run_all.py) runs on the output.

Equivalence argument: on an entry date the true nearest-delta strike for every parameter combination is in the
fetched set, so nearest-delta selection over the fetched subset gives the same legs as over the full chain.
It differs only if a planned leg has no valid quote (the guide's bid/ask filters would then pick the next
strike; here the fetch set may lack it). The build report counts how often that happens.

The `source` object needs: spx_daily(start, end), vix_daily(start, end), vix3m_daily(start, end),
universe_puts(date), put_quotes((expiry, strike), start, end). See qc_source.py (QuantConnect) and
mock_source.py (synthetic data, for tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vrp_data_layer import OPTION_COLS
from vrp_selection import pick_expiry, pick_strike
from vrp_stages import calendar_from_market

PLACEHOLDER_DELTA = -0.01   # on non-entry days only; selection never runs on those days (see assemble_options)
SETTLE_FROM_PRIOR_CLOSE = {pd.Timestamp("2018-12-05"): pd.Timestamp("2018-12-04")}  # National Day of Mourning (Bush): no SPX close that day

def read_cboe_close(path) -> pd.Series:
    """Cboe history CSV (VIX_History.csv, VIX3M_History.csv, ...): date first, CLOSE or last column."""
    df = pd.read_csv(path)
    c = next((c for c in df.columns if c.strip().upper() == "CLOSE"), df.columns[-1])
    return pd.Series(df[c].astype(float).to_numpy(), index=pd.to_datetime(df.iloc[:, 0]).astype("datetime64[ns]")
                     ).sort_index()


CBOE_VIX3M_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"


def vix3m_from_chunks(prefix: str = "vrp_vix3m_") -> pd.Series | None:
    """Read vrp_vix3m_1.py, vrp_vix3m_2.py, ... (made by qc/make_vix3m_py.py). QuantConnect's IDE only accepts
    .py/.ipynb/.cs/.html/.css files, so a CSV cannot be a project file; data lives in string constants instead."""
    import importlib
    parts, i = [], 1
    while True:
        try:
            parts.append(importlib.import_module(f"{prefix}{i}").CSV)
        except ImportError:
            break
        i += 1
    if not parts:
        return None
    rows = [ln.split(",") for ln in "\n".join(parts).splitlines() if ln and not ln.startswith("date")]
    return pd.Series([float(r[1]) for r in rows], index=pd.to_datetime([r[0] for r in rows]).astype("datetime64[ns]")).sort_index()


def load_vix3m(source, start, end, url: str = CBOE_VIX3M_URL, verbose: bool = True) -> pd.Series:
    """Try, in order: QuantConnect itself, the Cboe URL (needs internet in the notebook), vrp_vix3m_*.py chunks."""
    tried = []
    for name, fn in (("QuantConnect", lambda: source.vix3m_daily(start, end)),
                     ("Cboe URL", lambda: read_cboe_close(url)),
                     ("vrp_vix3m_*.py chunks", vix3m_from_chunks)):
        try:
            s = fn()
        except Exception as ex:
            tried.append(f"{name}: {type(ex).__name__}: {str(ex)[:80]}")
            continue
        if s is not None and len(s) > 500:
            if verbose:
                print(f"VIX3M from {name}: {s.index.min().date()}..{s.index.max().date()} ({len(s)} rows)")
            return s
        tried.append(f"{name}: nothing usable")
    raise RuntimeError("No VIX3M series. Tried -> " + " | ".join(tried) + ". Download VIX3M_History.csv from Cboe, "
                       "run `python qc/make_vix3m_py.py VIX3M_History.csv` locally, and add the vrp_vix3m_*.py files.")


def plan_dates_for(days: pd.DatetimeIndex, cut: pd.Timestamp) -> pd.DatetimeIndex:
    """Every date on which the engine can attempt an entry under ANY run window: the first trading day of each
    week, plus the first day of every run window (the engine forces an entry attempt there)."""
    periods = days.to_period("W")
    first = np.r_[True, np.asarray(periods[1:] != periods[:-1])]
    return days[first].union(pd.DatetimeIndex([days[0], cut]))


def plan_legs(source, plan_dates, cfg: dict, horizon_days: int = 120, verbose: bool = True):
    """Returns (legs, uni, info). legs: {(expiry, strike): first_entry_date}; uni: {date: universe puts}."""
    g = cfg["grid"]
    dtes = sorted({cfg["target_dte"], *g["dte"]})
    shorts = sorted({cfg["short_delta"], *g["short_delta"]})
    longs = sorted({cfg["long_delta"], *g["long_delta"]})
    legs, uni, empty, dte_hist = {}, {}, [], []
    for i, d in enumerate(plan_dates):
        u = source.universe_puts(d)
        if u is None or len(u) == 0:
            empty.append(d)
            continue
        u = u[(u["expiry"] > d) & (u["expiry"] <= d + pd.Timedelta(days=horizon_days))].dropna(subset=["delta"])
        u = u.sort_values(["expiry", "strike"]).reset_index(drop=True)
        if u.empty:
            empty.append(d)
            continue
        uni[d] = u
        for dte in dtes:
            e = pick_expiry(u, d, dte)
            if e is None:
                continue
            sub = u[u["expiry"] == e]
            if dte == cfg["target_dte"]:
                dte_hist.append((e - d).days)
            for target in [-x for x in shorts] + [-x for x in longs]:
                k = (pd.Timestamp(e), float(pick_strike(sub, target)["strike"]))
                legs.setdefault(k, d)
        if verbose and (i + 1) % 50 == 0:
            print(f"planned {i + 1}/{len(plan_dates)} dates, {len(legs)} legs so far")
    info = {"n_plan_dates": int(len(plan_dates)), "n_with_universe": len(uni), "empty_dates": [str(x.date()) for x in empty][:50],
            "first_universe_date": str(min(uni).date()) if uni else None,
            "n_legs": len(legs), "entry_dte": pd.Series(dte_hist).describe().round(2).to_dict() if dte_hist else {}}
    return legs, uni, info


def fetch_quotes(source, legs: dict, last_day, cache_dir, flush_every: int = 200, verbose: bool = True):
    """Quote snapshots (date, expiry, strike, bid, ask) for every planned leg; resumable via cache_dir."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    done_p = cache / "done.json"
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    key_s = lambda k: f"{k[0].date()}|{k[1]}"
    todo = [(k, d) for k, d in sorted(legs.items()) if key_s(k) not in done]
    buf, errors = [], []
    n_part = len(list(cache.glob("quotes_part_*.parquet")))

    def flush():
        nonlocal buf, n_part
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(cache / f"quotes_part_{n_part:05d}.parquet")
            n_part += 1
            buf = []
        done_p.write_text(json.dumps(sorted(done)))

    for i, (k, first) in enumerate(todo):
        try:
            q = source.put_quotes(k, first, min(k[0], pd.Timestamp(last_day)))
        except Exception as ex:                                  # keep going; report at the end
            errors.append((key_s(k), repr(ex)[:200]))
            continue
        if q is not None and len(q):
            q = q.assign(expiry=k[0], strike=k[1])
            buf.append(q[["date", "expiry", "strike", "bid", "ask"]])
        done.add(key_s(k))
        if len(buf) >= flush_every:
            flush()
            if verbose:
                print(f"fetched {i + 1}/{len(todo)} contracts")
    flush()
    if todo and len(errors) > 0.1 * len(todo):
        raise RuntimeError(f"{len(errors)} of {len(todo)} quote requests failed; first: {errors[:3]}")
    parts = sorted(cache.glob("quotes_part_*.parquet"))
    allq = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "expiry", "strike", "bid", "ask"])
    allq["date"] = pd.to_datetime(allq["date"]).astype("datetime64[ns]")
    allq["expiry"] = pd.to_datetime(allq["expiry"]).astype("datetime64[ns]")
    return allq.drop_duplicates(["date", "expiry", "strike"], keep="last"), errors


def assemble_options(quotes: pd.DataFrame, uni: dict, plan_dates, last_day, root: str = "SPXW") -> pd.DataFrame:
    """Rows in the vendor schema. delta is the real universe delta on plan (entry) days. On other days the engine
    only marks open spreads and never selects, so a placeholder delta is written (flagged in
    delta_is_placeholder). On a plan day a fetched contract that is missing from that day's universe gets NaN
    and is dropped by the loader's missing_delta rule (visible in the drop log)."""
    uni_all = pd.concat([u.assign(date=d)[["date", "expiry", "strike", "delta", "iv"]] for d, u in uni.items()],
                        ignore_index=True) if uni else pd.DataFrame(columns=["date", "expiry", "strike", "delta", "iv"])
    q = quotes[quotes["date"] <= pd.Timestamp(last_day)].merge(uni_all, on=["date", "expiry", "strike"], how="left")
    is_plan = q["date"].isin(pd.DatetimeIndex(plan_dates))
    q["delta_is_placeholder"] = ~is_plan
    q.loc[~is_plan, "delta"] = PLACEHOLDER_DELTA
    out = pd.DataFrame({"date": q["date"], "root": root, "expiry": q["expiry"], "strike": q["strike"], "cp": "P",
                        "bid": q["bid"], "ask": q["ask"], "delta": q["delta"], "iv": q["iv"], "volume": 0,
                        "open_interest": 0, "delta_is_placeholder": q["delta_is_placeholder"]})
    return out.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)


def build_dataset(source, cfg: dict, out_dir, history_start="2005-01-01", history_end=None, vix3m=None,
                  tbill=None, put_index=None, cache_dir=None, verbose: bool = True) -> dict:
    """End-to-end: write the processed-schema directory. Returns the build report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(history_end) if history_end else pd.Timestamp.today().normalize()

    spx = source.spx_daily(history_start, end)
    vix = source.vix_daily(history_start, end)
    if isinstance(vix3m, str):                       # a path or URL to a Cboe CSV
        vix3m = read_cboe_close(vix3m)
    elif vix3m is None:
        vix3m = load_vix3m(source, history_start, end, verbose=verbose)
    spx.reset_index(names="date").to_parquet(out / "spx_daily.parquet")
    pd.DataFrame({"date": vix.index, "close": vix.to_numpy()}).to_parquet(out / "vix.parquet")
    pd.DataFrame({"date": vix3m.index, "close": vix3m.to_numpy()}).to_parquet(out / "vix3m.parquet")
    for name, s, col in (("tbill", tbill, "rate"), ("put_index", put_index, "close")):
        if s is not None:
            pd.DataFrame({"date": s.index, col: s.to_numpy()}).to_parquet(out / f"{name}.parquet")

    market = {"spx_daily": spx, "vix": vix, "vix3m": vix3m, "spx_5min": None}
    cal = calendar_from_market(cfg, market)
    days, cut = cal["days"], cal["cut"]
    plan_dates = plan_dates_for(days, cut)
    if verbose:
        print(f"trading days {days[0].date()}..{days[-1].date()}, holdout from {cut.date()}, "
              f"{len(plan_dates)} plan dates")

    legs, uni, info = plan_legs(source, plan_dates, cfg, verbose=verbose)
    if not uni:
        raise RuntimeError("The universe returned no option data on any plan date.")
    first_data = pd.Timestamp(info["first_universe_date"])
    ts = pd.Timestamp(cfg["trade_start"]) if cfg.get("trade_start") else days[0]
    if first_data > ts + pd.Timedelta(days=10):
        raise RuntimeError(f"Option data starts {first_data.date()}, after trade_start/first tradable day "
                           f"{ts.date()}. Set trade_start: '{first_data.date()}' in the config (a flat period "
                           f"before the data would dilute every metric).")

    quotes, errors = fetch_quotes(source, legs, days[-1], cache_dir or (out / "qc_cache"), verbose=verbose)
    options = assemble_options(quotes, uni, plan_dates, days[-1], cfg["option_root"])
    for y, g in options.groupby(options["date"].dt.year):
        g.to_parquet(out / f"options_{y}.parquet")

    # PM-settled SPXW: official settlement = the SPX close on the expiry date.
    close = spx["close"]
    all_exps = pd.DatetimeIndex(sorted({k[0] for k in legs}))
    past = all_exps[all_exps <= days[-1]]
    settle = pd.Series({e: close.get(SETTLE_FROM_PRIOR_CLOSE.get(e, e), np.nan) for e in past}).dropna()
    exps = settle.index
    pd.DataFrame({"expiry": exps, "settle": settle.to_numpy()}).to_parquet(out / "settlement.parquet")

    entry_cov = _entry_coverage(legs, quotes)
    report = {**info, "n_quote_rows": int(len(quotes)), "n_quote_errors": len(errors), "errors_head": errors[:5],
              "entry_day_quote_coverage": entry_cov, "n_expiries_with_settlement": int(len(exps)),
              "n_expiries_without_settlement": int(len(past) - len(exps)),
              "n_expiries_after_last_day": int(len(all_exps) - len(past)),
              "first_tradable": str(days[0].date()), "holdout_start": str(cut.date()),
              "note": "Test 1.3 (PUT replica) is skipped on this route: no third-Friday ATM legs are fetched."}
    (out / "qc_build_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _entry_coverage(legs: dict, quotes: pd.DataFrame) -> float:
    """Share of planned legs that have a quote on their entry date (a missing entry quote means the guide's
    next-strike fallback would diverge from this dataset)."""
    if not legs:
        return float("nan")
    have = set(zip(quotes["date"], quotes["expiry"], quotes["strike"]))
    return float(np.mean([(d, k[0], k[1]) in have for k, d in legs.items()]))
