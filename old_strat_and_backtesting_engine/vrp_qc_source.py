"""QuantBookSource: the ONLY file that touches the QuantConnect API.

STATUS: written from QuantConnect's published docs; NOT run against the live API (no access from where it was
written). Everything else in qc/ is tested against a mock with the same five methods. Run `run_probe` first:
it checks each assumption below and prints what it finds before you start the long fetch.

Assumptions (all from the docs pages for Index Options Universes / Individual Contracts):
  * qb.add_index("SPX", Resolution.MINUTE).symbol and qb.add_index_option(spx_symbol, "SPXW") exist.
  * qb.history[OptionUniverse](option.symbol, start, end) yields one chain per day; each contract has
    .symbol.id (.date, .strike_price, .option_right), .greeks.delta, .implied_volatility, .price; chain.end_time.
  * qb.history(QuoteBar, contract_symbol, start, end, resolution) returns bid/ask bars after
    qb.add_index_option_contract(symbol, fill_forward=False).
Unverified: how daily bars are time-stamped (start vs end of day). calibrate_* below measures it.
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import pandas as pd

# SPX closes I know from memory; calibrate_index_shift() checks the downloaded bars against them and prints the match.
KNOWN_SPX_CLOSES = {"2019-12-31": 3230.78, "2020-03-13": 2711.02, "2020-03-16": 2386.13}


def _dt(x):
    return pd.Timestamp(x).to_pydatetime()


def _time_col(df: pd.DataFrame) -> str:
    for c in ("time", "endtime", "end_time"):
        if c in df.columns:
            return c
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime64"):
            return c
    raise KeyError(f"no time column among {list(df.columns)}")


def _find(df: pd.DataFrame, *names) -> str:
    low = {str(c).lower().replace("_", "").replace(".", ""): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f"none of {names} among {list(df.columns)}")


class QuantBookSource:
    def __init__(self, qb, index_ticker="SPX", option_ticker="SPXW", vix_ticker="VIX",
                 snapshot="16:00", quote_resolution="minute"):
        from AlgorithmImports import Resolution
        self.qb, self.snapshot, self.quote_resolution = qb, snapshot, quote_resolution
        self.spx = qb.add_index(index_ticker, Resolution.MINUTE).symbol
        self.vix = qb.add_index(vix_ticker, Resolution.DAILY).symbol
        self.option = qb.add_index_option(self.spx, option_ticker)
        self._sym: dict = {}
        self.index_shift = 0       # days added to daily-bar timestamps; set by calibrate_index_shift()
        self.universe_shift = 0    # days added to chain.end_time; set by calibrate_universe_shift()

    # ---- index series -------------------------------------------------------------------------------
    def _ohlc(self, symbol, start, end) -> pd.DataFrame:
        from AlgorithmImports import Resolution
        df = self.qb.history(symbol, _dt(start), _dt(end) + timedelta(days=1), Resolution.DAILY)
        if df is None or len(df) == 0:
            raise ValueError(f"no daily history for {symbol}")
        df = df.reset_index()
        t = pd.to_datetime(df[_time_col(df)]) + pd.Timedelta(days=self.index_shift)
        out = pd.DataFrame({c: df[c].astype(float).to_numpy() for c in ("open", "high", "low", "close")})
        out.index = pd.DatetimeIndex(t.dt.normalize()).astype("datetime64[ns]")
        return out[~out.index.duplicated()].sort_index()

    def spx_daily(self, start, end):
        return self._ohlc(self.spx, start, end)

    def vix_daily(self, start, end):
        return self._ohlc(self.vix, start, end)["close"]

    def vix3m_daily(self, start, end):
        """QuantConnect's US Indices page lists only SPX and VIX. Try VIX3M; None tells the caller to use a file."""
        from AlgorithmImports import Resolution
        try:
            sym = self.qb.add_index("VIX3M", Resolution.DAILY).symbol
            return self._ohlc(sym, start, end)["close"]
        except Exception:
            return None

    # ---- option universe ----------------------------------------------------------------------------
    def _raw_universe(self, d0, d1):
        from AlgorithmImports import OptionRight, OptionUniverse
        for chain in self.qb.history[OptionUniverse](self.option.symbol, _dt(d0), _dt(d1)):
            rows = []
            for c in chain:
                sid = c.symbol.id
                if sid.option_right != OptionRight.PUT:
                    continue
                key = (pd.Timestamp(sid.date).normalize(), float(sid.strike_price))
                self._sym[key] = c.symbol
                rows.append((key[0], key[1], float(c.greeks.delta), float(c.implied_volatility), float(c.price)))
            yield pd.Timestamp(chain.end_time), pd.DataFrame(rows, columns=["expiry", "strike", "delta", "iv", "price"])

    def universe_puts(self, d):
        d = pd.Timestamp(d).normalize()
        target_end = d - pd.Timedelta(days=self.universe_shift)   # shift=-1 -> chain.end_time == d + 1 day
        for end_time, df in self._raw_universe(target_end - pd.Timedelta(days=1),
                                            target_end + pd.Timedelta(days=1)):
            if (end_time + pd.Timedelta(days=self.universe_shift)).normalize() == d:
                return df[["expiry", "strike", "delta", "iv"]]
        return None

    # ---- quotes -------------------------------------------------------------------------------------
    def put_quotes(self, key, start, end):
        """Last quote at or before the snapshot time on each date, for one put contract."""
        from AlgorithmImports import QuoteBar, Resolution
        sym = self._sym.get((pd.Timestamp(key[0]).normalize(), float(key[1])))
        if sym is None:
            return None
        self.qb.add_index_option_contract(sym, fill_forward=False)
        res = Resolution.MINUTE if self.quote_resolution == "minute" else Resolution.HOUR
        df = self.qb.history(QuoteBar, sym, _dt(start), _dt(end) + timedelta(days=1), res)
        try:
            self.qb.remove_security(sym)
        except Exception:
            pass
        if df is None or len(df) == 0:
            return None
        df = df.reset_index()
        t = pd.to_datetime(df[_time_col(df)])
        bid, ask = df[_find(df, "bidclose")].astype(float), df[_find(df, "askclose")].astype(float)
        keep = (t.dt.time <= pd.Timestamp(self.snapshot).time()) & ask.notna()
        d = pd.DataFrame({"t": t[keep], "bid": bid[keep].fillna(0.0), "ask": ask[keep]})
        if d.empty:
            return None
        d["date"] = d["t"].dt.normalize().astype("datetime64[ns]")
        return d.sort_values("t").groupby("date", as_index=False).tail(1)[["date", "bid", "ask"]].reset_index(drop=True)

    # ---- calibration --------------------------------------------------------------------------------
    def calibrate_index_shift(self, verbose=True) -> dict:
        """Find the day offset that makes the downloaded SPX closes match the known closes above.

        Tolerance is 1.0 index point: QuantConnect's index values come from Cboe and can differ from the official
        close by cents, and my remembered closes may be off by the same. A wrong offset is off by tens of points
        (a typical daily move), so the winner must also beat the runner-up by more than 5 points.
        A missing date counts as an infinite error (max() alone would silently skip a NaN)."""
        from AlgorithmImports import Resolution
        raw = self.qb.history(self.spx, _dt("2019-12-27"), _dt("2020-03-21"), Resolution.DAILY).reset_index()
        t0 = pd.to_datetime(raw[_time_col(raw)])
        close = raw["close"].astype(float)
        known = {pd.Timestamp(d): v for d, v in KNOWN_SPX_CLOSES.items()}
        errs, detail = {}, {}
        for k in range(-3, 4):
            ser = pd.Series(close.to_numpy(), index=(t0 + pd.Timedelta(days=k)).dt.normalize())
            ser = ser[~ser.index.duplicated()]
            diffs = {d: (abs(float(ser[d]) - v) if d in ser.index else float("inf")) for d, v in known.items()}
            errs[k], detail[k] = max(diffs.values()), diffs
        order = sorted(errs, key=errs.get)
        k, runner = order[0], order[1]
        if errs[k] > 1.0 or errs[runner] - errs[k] < 5.0:
            raise RuntimeError(f"No timestamp shift cleanly reproduces the known SPX closes. Max error by shift: "
                               f"{ {j: round(e, 2) for j, e in errs.items()} }. Inspect "
                               f"qb.history(spx, ..., Resolution.DAILY) by hand.")
        self.index_shift = k
        if verbose:
            print(f"index timestamp shift = {k} day(s); max error vs known closes {errs[k]:.3f} points "
                  f"(runner-up shift {runner}: {errs[runner]:.1f})")
            for d, e in detail[k].items():
                print(f"   {d.date()}: known {known[d]:.2f}, downloaded diff {e:.3f}")
        return {"shift": k, "errors": errs}

    def calibrate_universe_shift(self, spx_close: pd.Series, n_probe=4, since="2014-01-01", verbose=True) -> dict:
        """Estimate SPX from deep in-the-money short-dated puts (S ~ K - price) on big-move days and see which
        date label (end_time or end_time - 1 day) matches the SPX close."""
        r = spx_close.pct_change().abs()
        probes = list(r[r.index >= pd.Timestamp(since)].nlargest(n_probe).index)
        votes = []
        for d in probes:
            for end_time, df in self._raw_universe(d - pd.Timedelta(days=1), d + pd.Timedelta(days=2)):
                e = end_time.normalize()
                near = df[(df["delta"] <= -0.95) & (df["price"] > 0) & (df["expiry"] >= e - pd.Timedelta(days=2))
                          & (df["expiry"] <= e + pd.Timedelta(days=8))]
                if near.empty:
                    continue
                s_est = float((near["strike"] - near["price"]).median())
                errs = {k: abs(s_est / spx_close.get(e + pd.Timedelta(days=k), np.nan) - 1) for k in (-1, 0)}
                errs = {k: v for k, v in errs.items() if np.isfinite(v)}
                if errs:
                    votes.append(min(errs, key=errs.get))
        if not votes:
            raise RuntimeError("Could not calibrate universe dates (no deep in-the-money probes found).")
        k = max(set(votes), key=votes.count)
        self.universe_shift = k
        if verbose:
            print(f"universe date shift = {k} (votes {votes} from probes {[str(p.date()) for p in probes]})")
        return {"shift": k, "votes": votes}


def run_probe(src: QuantBookSource, day="2024-01-04", n_quote_probes=3):
    """Check each API assumption on one recent day and estimate the fetch cost.
    Every step is isolated: a failure is printed and the probe continues, so one run shows everything
    that works and everything that does not. Send the whole output if anything says FAILED."""
    import traceback
    results, status = {}, {}

    def step(name, fn):
        print(f"\n=== {name}")
        try:
            results[name], status[name] = fn(), "OK"
            print(f"--- {name}: OK")
        except Exception as ex:
            results[name], status[name] = None, "FAILED"
            print(f"--- {name}: FAILED  {type(ex).__name__}: {str(ex)[:300]}")
            print("".join(traceback.format_exc().splitlines(True)[-6:]))
        return results[name]

    step("1) index timestamps", src.calibrate_index_shift)
    spx = step("2) SPX daily bars", lambda: src.spx_daily("2023-12-01", "2024-02-01"))
    if spx is not None:
        print(spx.tail(3))
    step("3) VIX daily bars", lambda: src.vix_daily("2023-12-01", "2024-02-01").tail(3))

    # The universe date alignment must be measured BEFORE universe_puts is used: it matches chains to dates with it.
    if spx is not None:
        step("4) universe date alignment", lambda: src.calibrate_universe_shift(
            src.spx_daily("2014-01-01", "2024-12-31")["close"]))
    else:
        print("\n=== 4) universe date alignment: skipped (needs step 2)")

    def universe():
        t0 = time.time()
        u = src.universe_puts(day)
        print(f"   {0 if u is None else len(u)} puts in {time.time() - t0:.1f}s (universe_shift = {src.universe_shift})")
        if u is None or len(u) == 0:
            raw = list(src._raw_universe(pd.Timestamp(day) - pd.Timedelta(days=3), pd.Timestamp(day) + pd.Timedelta(days=4)))
            print("   chains the API returned for that window (end_time, puts):",
                  [(str(et), len(df)) for et, df in raw])
            raise RuntimeError(f"Empty universe on {day}: see the chains listed above; check the date label shift, "
                               f"the SPXW subscription, or data permissions.")
        print(u.describe().round(3))
        return u
    u = step("5) option universe on the probe day", universe)

    def quotes():
        d = pd.Timestamp(day)
        e = u["expiry"][(u["expiry"] - d).dt.days >= 25].min()
        sub = u[u["expiry"] == e]
        for target in (-0.20, -0.05, -0.10)[:n_quote_probes]:
            row = sub.iloc[(sub["delta"] - target).abs().argmin()]
            t0 = time.time()
            q = src.put_quotes((e, float(row["strike"])), d, e)
            print(f"   {e.date()} K={row['strike']:.0f} delta={row['delta']:.3f}: "
                  f"{0 if q is None else len(q)} daily quotes in {time.time() - t0:.1f}s")
            if q is not None:
                print(q.head(3))
    if u is not None:
        step("6) bid/ask quotes for a few legs", quotes)
    else:
        print("\n=== 6) bid/ask quotes: skipped (needs step 5)")

    print("\nSUMMARY:", status)
    print(f"SHIFTS: index={src.index_shift} universe={src.universe_shift}")
    print("Estimated fetch time = (seconds per contract above) x (number of legs; the build reports it).")