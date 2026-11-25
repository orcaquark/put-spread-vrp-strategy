"""Data layer (guide Step 3): load, validate, align.

Processed-data schema (all files live in one data directory):

    options_<YYYY>.parquet   date, root, expiry, strike, cp, bid, ask, delta, iv, volume, open_interest
    spx_daily.parquet        date, open, high, low, close
    spx_5min.parquet         timestamp, price      (optional; first bar of each day = the open print)
    vix.parquet              date, close
    vix3m.parquet            date, close
    settlement.parquet       expiry, settle        (official settlement; SOQ for AM-settled expiries)
    tbill.parquet            date, rate            (optional; annual rate as a decimal)
    put_index.parquet        date, close           (optional; Cboe PUT index, for Test 1)
    spx_tr.parquet           date, close           (optional; SPX total return, for buy-and-hold)

CSV files with the same columns are also accepted (Cboe/FRED downloads need a rename of their header row).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OPTION_COLS = ["date", "root", "expiry", "strike", "cp", "bid", "ask", "delta", "iv",
               "volume", "open_interest"]
EMPTY_CHAIN = pd.DataFrame({c: pd.Series(dtype="float64") for c in OPTION_COLS})


def as_ns(x):
    """Coerce to tz-naive datetime64[ns] so dict keys and comparisons agree everywhere."""
    if isinstance(x, pd.Series):
        return pd.to_datetime(x).dt.tz_localize(None).astype("datetime64[ns]")
    idx = pd.DatetimeIndex(pd.to_datetime(x))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.astype("datetime64[ns]")


# --------------------------------------------------------------------------------------
# Option chains
# --------------------------------------------------------------------------------------
def clean_chain(raw: pd.DataFrame, preferred_root: str = "SPXW", puts_only: bool = True):
    """Apply the Step 3 checks. Returns (clean_df, drop_log_df, info).

    Rules, in order (every dropped row is logged by year and rule):
      1. missing bid or ask                 -> dropped
      2. negative bid or ask                -> dropped
      3. crossed market (ask < bid)         -> dropped
      4. missing delta                      -> dropped (strike selection needs it)
      5. duplicate (date, expiry, strike, cp) across roots -> keep preferred root
    NOT dropped here: bid == 0. A long leg may have bid == 0; selection requires bid > 0 for short legs.
    No forward-filling of quotes happens anywhere.
    """
    df = raw.copy()
    missing = [c for c in OPTION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"option file is missing columns: {missing}")
    df["date"] = as_ns(df["date"])
    df["expiry"] = as_ns(df["expiry"])
    df["cp"] = df["cp"].astype(str).str.upper().str[0]
    if puts_only:
        df = df[df["cp"] == "P"]
    total_by_year = df.groupby(df["date"].dt.year).size()

    # Put-delta sign convention: the guide assumes puts carry negative delta.
    puts = df["cp"] == "P"
    pos_share = (df.loc[puts, "delta"] > 0).mean() if puts.any() else 0.0
    info = {"delta_sign_flipped": False, "non_preferred_root_expiries": []}
    if pos_share >= 0.99:
        df.loc[puts, "delta"] = -df.loc[puts, "delta"]
        info["delta_sign_flipped"] = True
        warnings.warn("Put deltas were positive in the vendor file; negated to the guide's convention.")
    elif pos_share > 0.01:
        raise ValueError("Put deltas have mixed signs; fix the vendor convention before loading.")

    log = []

    def drop(mask: pd.Series, rule: str):
        nonlocal df
        if mask.any():
            for y, n in df.loc[mask, "date"].dt.year.value_counts().items():
                log.append({"year": int(y), "rule": rule, "n_dropped": int(n)})
            df = df[~mask]

    drop(df["bid"].isna() | df["ask"].isna(), "missing_quote")
    drop((df["bid"] < 0) | (df["ask"] < 0), "negative_quote")
    drop(df["ask"] < df["bid"], "crossed_market")
    drop(df["delta"].isna(), "missing_delta")

    # One root per (date, expiry, strike, cp): prefer the PM-settled root.
    df = df.assign(_pref=(df["root"] != preferred_root).astype(int))
    df = df.sort_values(["date", "expiry", "strike", "cp", "_pref"])
    dup = df.duplicated(["date", "expiry", "strike", "cp"], keep="first")
    drop(dup, "duplicate_across_roots")
    nonpref = df.loc[df["root"] != preferred_root, "expiry"].drop_duplicates()
    info["non_preferred_root_expiries"] = sorted(nonpref.dt.strftime("%Y-%m-%d"))
    df = df.drop(columns="_pref").reset_index(drop=True)

    drop_log = pd.DataFrame(log, columns=["year", "rule", "n_dropped"])
    if len(drop_log):
        drop_log = drop_log.groupby(["year", "rule"], as_index=False)["n_dropped"].sum()
    drop_log = drop_log.merge(total_by_year.rename("n_rows").rename_axis("year").reset_index(),
                              on="year", how="right").fillna({"n_dropped": 0})
    return df, drop_log, info


class ChainStore:
    """Callable chain_for(d) -> that day's quotes (an empty frame if the vendor has no data that day)."""

    def __init__(self, df: pd.DataFrame):
        self._d = {k: g.reset_index(drop=True) for k, g in df.groupby("date", sort=True)}
        self.dates = pd.DatetimeIndex(sorted(self._d))

    def __call__(self, d) -> pd.DataFrame:
        return self._d.get(d, EMPTY_CHAIN)


def load_options(data_dir, preferred_root="SPXW", puts_only=True):
    """Clean each yearly file separately (bounded memory), then concatenate."""
    files = sorted(Path(data_dir).glob("options_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no options_<YYYY>.parquet files in {data_dir}")
    frames, logs, roots = [], [], set()
    for fp in files:
        df, log, info = clean_chain(pd.read_parquet(fp), preferred_root, puts_only)
        frames.append(df)
        logs.append(log)
        roots.update(info["non_preferred_root_expiries"])
    drop_log = pd.concat(logs, ignore_index=True)
    return pd.concat(frames, ignore_index=True), drop_log, sorted(roots)


# --------------------------------------------------------------------------------------
# Index series, settlement
# --------------------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _find(data_dir, stem):
    for ext in (".parquet", ".csv"):
        p = Path(data_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _series(data_dir, stem, value_col, required=True):
    p = _find(data_dir, stem)
    if p is None:
        if required:
            raise FileNotFoundError(f"{stem}.parquet/.csv not found in {data_dir}")
        return None
    df = _read(p)
    s = pd.Series(df[value_col].to_numpy(dtype=float), index=as_ns(df["date"]), name=stem)
    return s[~s.index.duplicated()].sort_index()


def load_market(data_dir) -> dict:
    d = Path(data_dir)
    p = _find(d, "spx_daily")
    if p is None:
        raise FileNotFoundError(f"spx_daily.parquet/.csv not found in {d}")
    spx = _read(p)
    spx.index = as_ns(spx.pop("date"))
    spx = spx.sort_index().astype(float)
    m = {"spx_daily": spx,
         "vix": _series(d, "vix", "close"),
         "vix3m": _series(d, "vix3m", "close"),
         "tbill": _series(d, "tbill", "rate", required=False),
         "put_index": _series(d, "put_index", "close", required=False),
         "spx_tr": _series(d, "spx_tr", "close", required=False)}
    p5 = _find(d, "spx_5min")
    if p5 is not None:
        df = _read(p5)
        m["spx_5min"] = pd.Series(df["price"].to_numpy(dtype=float), index=as_ns(df["timestamp"]))
    else:
        m["spx_5min"] = None
    ps = _find(d, "settlement")
    if ps is None:
        raise FileNotFoundError(f"settlement.parquet/.csv not found in {d}")
    st = _read(ps)
    m["settle"] = pd.Series(st["settle"].to_numpy(dtype=float), index=as_ns(st["expiry"])).sort_index()
    return m


def align(market: dict) -> pd.DatetimeIndex:
    """Single trading-day index = days present in SPX, VIX and VIX3M. Index series are NOT forward-filled."""
    idx = market["spx_daily"].index.intersection(market["vix"].index).intersection(market["vix3m"].index)
    return idx.sort_values()


# --------------------------------------------------------------------------------------
# Realized variance (daily, NOT annualized)
# --------------------------------------------------------------------------------------
def rv_5min(px: pd.Series, include_overnight: bool = False) -> pd.Series:
    """Sum of squared 5-minute log returns per day.

    px must contain, per day, the open print followed by the 5-minute bars. The first return of each day
    (open vs previous close) is the overnight gap: excluded by default (the guide's definition).
    """
    lr = np.log(px).diff()
    day = pd.Series(px.index.normalize(), index=px.index)
    first = day.ne(day.shift())
    intraday = (lr.where(~first) ** 2).groupby(day.to_numpy()).sum(min_count=1)
    if include_overnight:
        intraday = intraday + (lr.where(first) ** 2).groupby(day.to_numpy()).sum(min_count=1).fillna(0.0)
    intraday.index = pd.DatetimeIndex(intraday.index)
    return intraday


def parkinson_var(ohlc: pd.DataFrame) -> pd.Series:
    return np.log(ohlc["high"] / ohlc["low"]) ** 2 / (4 * np.log(2))


def garman_klass_var(ohlc: pd.DataFrame) -> pd.Series:
    return (0.5 * np.log(ohlc["high"] / ohlc["low"]) ** 2
            - (2 * np.log(2) - 1) * np.log(ohlc["close"] / ohlc["open"]) ** 2)


def realized_variance(market: dict, days: pd.DatetimeIndex, cfg: dict, include_overnight=None) -> pd.Series:
    est = cfg["rv"]["estimator"]
    ovn = cfg["rv"]["include_overnight"] if include_overnight is None else include_overnight
    if est == "5min":
        if market.get("spx_5min") is None:
            raise FileNotFoundError("rv.estimator is '5min' but spx_5min.parquet is missing; "
                                    "set rv.estimator to parkinson or garman_klass for the daily fallback.")
        rv = rv_5min(market["spx_5min"], ovn)
    elif est == "parkinson":
        rv = parkinson_var(market["spx_daily"])
    elif est == "garman_klass":
        rv = garman_klass_var(market["spx_daily"])
    else:
        raise ValueError(f"unknown rv.estimator {est!r}")
    return rv.reindex(days)
