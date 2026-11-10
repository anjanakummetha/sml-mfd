"""
Disagreement-to-Alpha backtester (Task 7, Fall 2026).

Run from the repository root:

    python -m src.backtester config/backtest/baseline.json [more configs ...] [--no-plot]

Results go to results/backtest/<config name>/. Paths inside configs are relative to the repository root.

Input: a stock-month panel where each row is a stock at a formation month `eom`,
with signal columns known at `eom`, the return realized over the NEXT month
(e.g. `actual_excess_return` == JKP `ret_exc_lead1m`), and market cap `me` at `eom`.

Each strategy:
  1. builds one score per stock-month from one or more signal columns
     (each standardized cross-sectionally within the month, then combined with weights),
  2. buys the stocks whose score percentile is inside `long_pct` and shorts those inside
     `short_pct` (percentiles run 0 = lowest score .. 100 = highest score),
  3. holds each month's portfolio ("cohort") for `holding_months` months.

Holding periods (overlapping portfolios, Jegadeesh-Titman style):
  With holding_months = K, a new cohort is formed every month with 1/K of the capital and
  held for K months, so in any month K cohorts are live. The strategy return in a month is
  the average of the live cohorts' returns. Within a cohort the formation weights are kept
  (renormalized over stocks that still have a return that month). K = 1 is the plain
  monthly-rebalanced sort.
  ramp_up = "drop": months before all K cohorts are live are left out (literature standard).
  ramp_up = "cash": they are kept; sleeves not yet invested earn 0.

All output series are indexed by the month the return is EARNED (formation month + 1 for
the first holding month), so Fama-French factors are matched to the same month.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

MonthEnd = pd.offsets.MonthEnd
REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    # {column: weight}. Each column is standardized within month, then summed with weights.
    signal: dict
    # "rank" (percentile rank, default), "zscore", or "raw"
    standardize: str = "rank"
    # Percentile ranges of the combined score. Low disagreement = low percentiles.
    long_pct: list | None = field(default_factory=lambda: [0, 10])
    short_pct: list | None = field(default_factory=lambda: [90, 100])
    # Shortcut: tail_pct=10 means long [0,10], short [90,100] (overrides long/short_pct)
    tail_pct: float | None = None
    weighting: str = "equal"            # "equal" or "value"
    holding_months: int = 1             # K: each cohort is held K months (overlapping)
    ramp_up: str = "drop"               # "drop" or "cash" (see module docstring)
    n_buckets: int = 10                 # for the full bucket table
    start: str | None = None            # first formation month, e.g. "2005-01"
    end: str | None = None              # last formation month
    months_file: str | None = None      # CSV with an `eom` column: keep only those formation months
    drop_size_pct: float = 0.0          # drop the smallest X% of stocks by `me` each month
    nw_lags: int = 6                    # Newey-West lags for t-stats
    # How tied scores are ranked: "average" (default) or "min" (ties pushed to the lower
    # percentile, which reproduces pd.qcut as used in last year's Task 7 code)
    tie_method: str = "average"

    def __post_init__(self):
        if self.tail_pct is not None:
            self.long_pct = [0, self.tail_pct]
            self.short_pct = [100 - self.tail_pct, 100]
        checks = {
            "weighting": ("equal", "value"),
            "standardize": ("rank", "zscore", "raw"),
            "ramp_up": ("drop", "cash"),
            "tie_method": ("average", "min"),
        }
        for k, ok in checks.items():
            if getattr(self, k) not in ok:
                raise ValueError(f"{self.name}: {k} must be one of {ok}")
        if self.long_pct is None and self.short_pct is None:
            raise ValueError(f"{self.name}: need at least one of long_pct / short_pct")
        self.holding_months = int(self.holding_months)
        if self.holding_months < 1:
            raise ValueError(f"{self.name}: holding_months must be >= 1")
        if not self.signal:
            raise ValueError(f"{self.name}: signal needs at least one column")


def expand_strategies(cfg: dict) -> list[Strategy]:
    """Merge `defaults` into each strategy and expand any `sweep` (Cartesian product)."""
    defaults = cfg.get("defaults", {})
    out = []
    for s in cfg["strategies"]:
        s = {**defaults, **s}
        sweep = s.pop("sweep", None) or {}
        if not sweep:
            out.append(Strategy(**s))
            continue
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            v = dict(zip(keys, combo))
            tag = "__".join(f"{k}={_fmt(x)}" for k, x in v.items())
            out.append(Strategy(**{**s, **v, "name": f"{s['name']}__{tag}"}))
    return out


def _fmt(x):
    return "-".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def month_index(dates: pd.Series) -> pd.Series:
    """Month-end dates -> integer month counter (so month arithmetic is just +/-)."""
    return dates.dt.year * 12 + dates.dt.month - 1


def index_to_month(idx) -> pd.DatetimeIndex:
    idx = np.asarray(idx, dtype=int)
    first = pd.to_datetime(pd.DataFrame({"year": idx // 12, "month": idx % 12 + 1, "day": 1}))
    return pd.DatetimeIndex(first + MonthEnd(0))


def load_panel(path, id_col="id", date_col="eom", ret_col="actual_excess_return", me_col="me",
               verbose=True):
    """Rows without a next-month return are KEPT: they are part of the formation universe
    (you can't know at formation that a stock will delist); they just contribute no return."""
    df = pd.read_csv(path)
    rename = {id_col: "id", date_col: "eom", ret_col: "ret"}
    if me_col:
        rename[me_col] = "me"
    df = df.rename(columns=rename)
    df["eom"] = pd.to_datetime(df["eom"]) + MonthEnd(0)
    df["fm"] = month_index(df["eom"])
    if verbose:
        print(f"Loaded {path}: {len(df):,} rows, {df.ret.isna().sum():,} without a next-month return, "
              f"{df.eom.nunique()} formation months ({df.eom.min():%Y-%m} to {df.eom.max():%Y-%m})")
    return df


def load_ff_factors(factors_dir) -> pd.DataFrame:
    """FF5 + Momentum, monthly, in DECIMALS, indexed by month-end."""
    d = Path(factors_dir)

    def read(fn, skip):
        raw = pd.read_csv(d / fn, skiprows=skip)
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.rename(columns={raw.columns[0]: "ym"})
        raw["ym"] = raw["ym"].astype(str).str.strip()
        raw = raw[raw["ym"].str.fullmatch(r"\d{6}")].copy()   # monthly block only
        for c in raw.columns[1:]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce") / 100.0
        raw["month"] = pd.to_datetime(raw["ym"], format="%Y%m") + MonthEnd(0)
        return raw.drop(columns="ym").set_index("month")

    ff5 = read("F-F_Research_Data_5_Factors_2x3.csv", 3)
    mom = read("F-F_Momentum_Factor.csv", 13)
    return ff5.join(mom[["Mom"]], how="inner")


FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}

# --------------------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------------------

@dataclass
class Result:
    strategy: Strategy
    monthly: pd.DataFrame     # index = month return is earned
    buckets: pd.DataFrame     # month, bucket, ret, n
    ic: pd.Series             # per formation month: rank IC of score vs next-month return
    universe: pd.DataFrame    # per formation month: stock counts
    meta: dict                # bookkeeping used by sanity_checks()


def formation_panel(panel: pd.DataFrame, s: Strategy, base_dir: Path = REPO_ROOT) -> pd.DataFrame:
    """Formation universe for one strategy with its score, percentile and bucket."""
    cols = ["id", "eom", "fm", "ret"] + list(s.signal) + (["me"] if "me" in panel else [])
    missing = [c for c in s.signal if c not in panel.columns]
    if missing:
        raise KeyError(f"{s.name}: signal columns not in data: {missing}")
    df = panel[list(dict.fromkeys(cols))].dropna(subset=list(s.signal))
    if s.start:
        df = df[df.eom >= pd.Timestamp(s.start) + MonthEnd(0)]
    if s.end:
        df = df[df.eom <= pd.Timestamp(s.end) + MonthEnd(0)]
    if s.months_file:
        keep = pd.to_datetime(pd.read_csv(base_dir / s.months_file)["eom"]) + MonthEnd(0)
        df = df[df.eom.isin(set(keep))]
    if s.weighting == "value" and "me" not in df:
        raise ValueError(f"{s.name}: value weighting needs a market-cap column")
    if s.drop_size_pct > 0:
        df = df[df.groupby("fm")["me"].rank(pct=True) > s.drop_size_pct / 100.0]

    score = pd.Series(0.0, index=df.index)
    g = df.groupby("fm")
    for col, w in s.signal.items():
        if s.standardize == "rank":
            z = g[col].rank(pct=True)
        elif s.standardize == "zscore":
            z = (df[col] - g[col].transform("mean")) / g[col].transform("std")
        else:
            z = df[col]
        score = score + w * z
    df = df.assign(score=score)
    df["pct"] = df.groupby("fm")["score"].rank(pct=True, method=s.tie_method) * 100
    df["bucket"] = np.clip(np.ceil(df["pct"] * s.n_buckets / 100), 1, s.n_buckets).astype(int)
    return df


def _in_range(pct: pd.Series, rng) -> pd.Series:
    lo, hi = rng
    return ((pct > lo) & (pct <= hi)) if lo > 0 else (pct <= hi)


def _formation_weights(sel: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Cohort weights fixed at formation (sum to 1 within each formation month)."""
    if weighting == "value":
        sel = sel.dropna(subset=["me"])
        w = sel["me"] / sel.groupby("fm")["me"].transform("sum")
    else:
        w = 1.0 / sel.groupby("fm")["id"].transform("count")
    return pd.DataFrame({"fm": sel.fm.values, "id": sel.id.values, "w": w.values})


def _cohort_returns(w: pd.DataFrame, rets: pd.DataFrame, K: int) -> pd.DataFrame:
    """Return of each cohort (formation month fm) in each of its K holding months rm.
    Stock return earned in month rm = `ret` on that stock's row at formation month rm - 1."""
    parts = []
    for h in range(1, K + 1):
        x = w.assign(src=w.fm + h - 1).merge(rets, on=["id", "src"], how="inner")
        x["rm"] = x.fm + h
        parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    x["wr"] = x.w * x.ret
    g = x.groupby(["fm", "rm"])
    return pd.DataFrame({"ret": g.wr.sum() / g.w.sum(), "n": g.size(), "w_used": g.w.sum()}).reset_index()


def _combine(cr: pd.DataFrame, K: int, ramp_up: str):
    g = cr.groupby("rm")["ret"]
    r = g.sum() / K if ramp_up == "cash" else g.mean()
    return r, g.size()


def _book(w: pd.DataFrame, K: int, ramp_up: str) -> pd.DataFrame:
    """Aggregate holdings in each return month rm: the live cohorts' formation weights."""
    book = pd.concat([w.assign(rm=w.fm + h) for h in range(1, K + 1)], ignore_index=True)
    if ramp_up == "cash":
        book["w"] = book.w / K
    else:
        book["w"] = book.w / book.groupby("rm")["fm"].transform("nunique")
    return book.groupby(["rm", "id"], as_index=False)["w"].sum()


def _turnover(book: pd.DataFrame) -> pd.Series:
    """One-sided turnover of the aggregate book: 0.5 * sum |w_t - w_{t-1}| (ignores drift)."""
    w = book.pivot_table(index="rm", columns="id", values="w", fill_value=0.0)
    return 0.5 * w.diff().abs().sum(axis=1).iloc[1:]


def run_strategy(panel: pd.DataFrame, s: Strategy, base_dir: Path = REPO_ROOT) -> Result:
    K = s.holding_months
    rets = panel.loc[panel.ret.notna(), ["id", "fm", "ret"]].rename(columns={"fm": "src"})
    df = formation_panel(panel, s, base_dir)

    legs = {side: _in_range(df.pct, rng) for side, rng in (("long", s.long_pct), ("short", s.short_pct))
            if rng is not None}
    cols, books, cohort_tables = {}, {}, {}
    n_cohorts = None
    for side, mask in legs.items():
        w = _formation_weights(df[mask], s.weighting)
        cr = _cohort_returns(w, rets, K)
        r, n_live = _combine(cr, K, s.ramp_up)
        book = _book(w, K, s.ramp_up)
        cols[f"{side}_ret"] = r
        cols[f"n_{side}"] = book[book.w > 0].groupby("rm").size()
        cols[f"turnover_{side}"] = _turnover(book)
        books[side], cohort_tables[side] = book, cr
        n_cohorts = n_live if n_cohorts is None else n_cohorts
    monthly = pd.DataFrame(cols)
    monthly["n_cohorts"] = n_cohorts
    monthly = monthly[monthly[[c for c in monthly if c.endswith("_ret")]].notna().any(axis=1)]
    # the track record ends one month after the last formation month (no tail of winding-down cohorts)
    last_return_month = int(df.fm.max()) + 1
    monthly = monthly[monthly.index <= last_return_month]

    # ramp-up: drop months before all K cohorts are live
    dropped = 0
    if s.ramp_up == "drop" and K > 1 and (monthly.n_cohorts >= K).any():
        first_full = monthly.index[monthly.n_cohorts >= K].min()
        dropped = int((monthly.index < first_full).sum())
        monthly = monthly[monthly.index >= first_full]

    if "long_ret" in monthly and "short_ret" in monthly:
        monthly["ls_ret"] = monthly["long_ret"] - monthly["short_ret"]
        monthly["strategy_ret"] = monthly["ls_ret"]
    elif "long_ret" in monthly:
        monthly["strategy_ret"] = monthly["long_ret"]
    else:
        monthly["strategy_ret"] = -monthly["short_ret"]

    bucket_rows = []
    for b in range(1, s.n_buckets + 1):
        w = _formation_weights(df[df.bucket == b], s.weighting)
        cr = _cohort_returns(w, rets, K)
        r, _ = _combine(cr, K, s.ramp_up)
        n = cr.groupby("rm")["n"].mean()
        bucket_rows.append(pd.DataFrame({"rm": r.index, "bucket": b, "ret": r.values,
                                         "n": n.reindex(r.index).values}))
    buckets = pd.concat(bucket_rows, ignore_index=True)
    buckets = buckets[buckets.rm.isin(monthly.index)]

    # diagnostics
    has_ret = df[df.ret.notna()]
    ic = has_ret.groupby("fm")[["score", "ret"]].apply(lambda g: g["score"].corr(g["ret"], method="spearman"))
    universe = pd.DataFrame({"n_stocks": df.groupby("fm").size(),
                             "n_with_return": has_ret.groupby("fm").size()})
    for side, mask in legs.items():
        universe[f"n_{side}"] = df[mask].groupby("fm").size()
    universe = universe.fillna(0)

    meta = {"min_holding_lag": min(int((ct.rm - ct.fm).min()) for ct in cohort_tables.values()),
            "max_holding_lag": max(int((ct.rm - ct.fm).max()) for ct in cohort_tables.values()),
            "warmup_months_dropped": dropped,
            "n_formation_months": int(df.fm.nunique()),
            "last_formation_month": index_to_month([int(df.fm.max())])[0],
            "book_weight_sums": {side: b.groupby("rm").w.sum() for side, b in books.items()}}

    # integer months -> month-end dates
    monthly.index = index_to_month(monthly.index)
    monthly.index.name = "month"
    buckets["month"] = index_to_month(buckets.pop("rm"))
    ic.index = index_to_month(ic.index)
    universe.index = index_to_month(universe.index)
    universe.index.name = "formation_month"
    for side, sums in meta["book_weight_sums"].items():
        sums.index = index_to_month(sums.index)
    return Result(s, monthly, buckets[["month", "bucket", "ret", "n"]], ic, universe, meta)

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _nw_t(x: pd.Series, lags: int) -> float:
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def perf_metrics(r: pd.Series, nw_lags=6) -> dict:
    r = r.dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    cagr = wealth.iloc[-1] ** (12 / n) - 1 if wealth.iloc[-1] > 0 else np.nan
    max_dd = dd.min()
    return {
        "n_months": n,
        "mean_monthly": mu,
        "t_stat": mu / (sd / np.sqrt(n)),
        "t_stat_nw": _nw_t(r, nw_lags),
        "ann_return": mu * 12,
        "cagr": cagr,
        "ann_vol": sd * np.sqrt(12),
        "sharpe": mu / sd * np.sqrt(12) if sd > 0 else np.nan,
        "sortino": mu * 12 / (downside * np.sqrt(12)) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "win_rate": (r > 0).mean(),
        "best_month": r.max(),
        "worst_month": r.min(),
        "skew": r.skew(),
        "excess_kurtosis": r.kurt(),
    }


def factor_regressions(r: pd.Series, factors: pd.DataFrame, nw_lags=6) -> pd.DataFrame:
    """Regress strategy returns on factors of the SAME month (series is indexed by return month)."""
    y = r.dropna()
    rows = []
    for model, cols in FACTOR_MODELS.items():
        X = factors[cols].reindex(y.index)
        ok = X.notna().all(axis=1)
        if ok.sum() <= len(cols) + 2:
            continue
        res = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        row = {"model": model, "n": int(res.nobs), "alpha": res.params["const"],
               "alpha_t_nw": res.tvalues["const"], "r2": res.rsquared}
        for c in cols:
            row[f"beta_{c}"] = res.params[c]
            row[f"t_{c}"] = res.tvalues[c]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(res: Result, factors: pd.DataFrame | None) -> tuple[dict, pd.DataFrame | None]:
    s, monthly, buckets = res.strategy, res.monthly, res.buckets
    m = perf_metrics(monthly["strategy_ret"], s.nw_lags)
    for side in ("long", "short"):
        if f"{side}_ret" in monthly:
            m[f"{side}_mean_monthly"] = monthly[f"{side}_ret"].mean()
            m[f"avg_n_{side}"] = monthly[f"n_{side}"].mean()
            m[f"avg_turnover_{side}"] = monthly[f"turnover_{side}"].mean()
    bmeans = buckets.groupby("bucket")["ret"].mean()
    m["bucket_monotonicity"] = pd.Series(bmeans.index).corr(pd.Series(bmeans.values), method="spearman")
    m["bucket_low_minus_high"] = bmeans.iloc[0] - bmeans.iloc[-1]
    ic = res.ic.dropna()
    m["mean_rank_ic"] = ic.mean()
    m["rank_ic_t"] = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 else np.nan

    fr = factor_regressions(monthly["strategy_ret"], factors, s.nw_lags) if factors is not None else None
    if fr is not None and len(fr):
        for _, row in fr.iterrows():
            m[f"alpha_{row.model}"] = row.alpha
            m[f"alpha_{row.model}_t"] = row.alpha_t_nw
        fr.insert(0, "strategy", s.name)

    head = {"strategy": s.name, "signal": json.dumps(s.signal), "standardize": s.standardize,
            "long_pct": _fmt(s.long_pct) if s.long_pct else "", "short_pct": _fmt(s.short_pct) if s.short_pct else "",
            "weighting": s.weighting, "holding_months": s.holding_months, "ramp_up": s.ramp_up}
    return {**head, **m}, fr


def sanity_checks(res: Result) -> list[dict]:
    """Quick checks that the engine did what the settings asked. Each: check, ok, detail."""
    s, u, meta, mo = res.strategy, res.universe, res.meta, res.monthly
    out = []

    def add(name, ok, detail):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    add("No look-ahead", meta["min_holding_lag"] >= 1,
        f"every return is earned 1-{meta['max_holding_lag']} months after the portfolio is formed")
    add("Holding period", meta["max_holding_lag"] == s.holding_months,
        f"longest hold {meta['max_holding_lag']} months (setting: {s.holding_months})")
    for side, rng in (("long", s.long_pct), ("short", s.short_pct)):
        if rng is None:
            continue
        want = rng[1] - rng[0]
        got = (u[f"n_{side}"] / u["n_stocks"]).mean() * 100
        add(f"{side.title()} leg size", abs(got - want) < max(1.0, 0.1 * want),
            f"{got:.2f}% of stocks on average (setting: {want:g}%)")
        sums = meta["book_weight_sums"][side]
        target = (mo["n_cohorts"] / s.holding_months) if s.ramp_up == "cash" else pd.Series(1.0, index=mo.index)
        err = (sums.reindex(mo.index) - target).abs().max()
        add(f"{side.title()} weights", err < 1e-9,
            f"book weights sum to {'1' if s.ramp_up == 'drop' else 'live cohorts / K'} every month "
            f"(max error {err:.1e})")
    full = (mo["n_cohorts"] >= s.holding_months).mean() * 100
    add("Cohorts live", s.ramp_up == "cash" or full == 100,
        f"{full:.0f}% of months have all {s.holding_months} cohorts; "
        f"{meta['warmup_months_dropped']} ramp-up months dropped")
    last = meta["last_formation_month"] + MonthEnd(1)
    add("Return window", mo.index.max() <= last,
        f"returns run {mo.index.min():%Y-%m} to {mo.index.max():%Y-%m}; last formation month "
        f"{meta['last_formation_month']:%Y-%m}")
    cov = (u["n_with_return"] / u["n_stocks"]).mean() * 100
    add("Return coverage", cov > 95, f"{cov:.1f}% of formation stocks have a next-month return")
    # two separate code paths should agree: bottom-minus-top bucket vs the long/short legs
    tail = 100 / s.n_buckets
    if s.long_pct == [0, tail] and s.short_pct == [100 - tail, 100] and "ls_ret" in mo:
        b = res.buckets.pivot(index="month", columns="bucket", values="ret")
        diff = ((b[1] - b[s.n_buckets]) - mo["ls_ret"]).abs().max()
        add("Buckets match legs", diff < 1e-9, f"bucket 1 - bucket {s.n_buckets} equals long - short "
            f"(max difference {diff:.1e})")
    return out

# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def run_config(config_path, out_root="results/backtest", plot=True):
    config_path = Path(config_path)
    base = REPO_ROOT                                     # config paths are relative to the repo root
    cfg = json.loads(config_path.read_text())
    strategies = expand_strategies(cfg)

    data = cfg["data"]
    panel = load_panel(base / data["path"], data.get("id_col", "id"), data.get("date_col", "eom"),
                       data.get("ret_col", "actual_excess_return"), data.get("me_col", "me"))
    factors = load_ff_factors(base / cfg["factors_dir"]) if cfg.get("factors_dir") else None

    out = base / out_root / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    summaries, monthly_all, bucket_all, fr_all = [], [], [], []
    for s in strategies:
        res = run_strategy(panel, s, base)
        summ, fr = summarize(res, factors)
        failed = [c["check"] for c in sanity_checks(res) if not c["ok"]]
        summaries.append(summ)
        monthly_all.append(res.monthly.reset_index().assign(strategy=s.name))
        bucket_all.append(res.buckets.assign(strategy=s.name))
        if fr is not None:
            fr_all.append(fr)
        print(f"  {s.name:<55} mean {summ['mean_monthly']*100:6.2f}%/mo  t={summ['t_stat']:5.2f}  "
              f"Sharpe {summ['sharpe']:5.2f}  MaxDD {summ['max_drawdown']*100:6.1f}%"
              + (f"  FF6 a={summ['alpha_FF6']*100:5.2f}% (t={summ['alpha_FF6_t']:.2f})" if "alpha_FF6" in summ else "")
              + (f"  CHECKS FAILED: {failed}" if failed else ""))

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(monthly_all).to_csv(out / "monthly_returns.csv", index=False)
    bucket_tbl = pd.concat(bucket_all)
    bucket_tbl.to_csv(out / "bucket_returns_monthly.csv", index=False)
    (bucket_tbl.groupby(["strategy", "bucket"])["ret"].mean().unstack("bucket")
        .to_csv(out / "bucket_means.csv"))
    if fr_all:
        pd.concat(fr_all).to_csv(out / "factor_regressions.csv", index=False)
    (out / "config_used.json").write_text(json.dumps(
        {**cfg, "expanded_strategies": [asdict(s) for s in strategies]}, indent=2))
    if plot:
        _plot_cumulative(pd.concat(monthly_all), out / "cumulative_returns.png")
    print(f"Wrote results to {out}")
    return summary


def _plot_cumulative(monthly_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, g in monthly_all.groupby("strategy", sort=False):
        g = g.sort_values("month")
        ax.plot(pd.to_datetime(g.month), (1 + g.strategy_ret.fillna(0)).cumprod(), label=name, lw=1.3)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_title("Growth of $1 (month the return is earned)")
    ax.set_ylabel("Cumulative wealth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Disagreement portfolio-sort backtester (Task 7)")
    parser.add_argument("configs", nargs="+", type=Path, help="Config files, e.g. config/backtest/baseline.json")
    parser.add_argument("--out", default="results/backtest", help="Output folder, relative to the repo root")
    parser.add_argument("--no-plot", action="store_true", help="Skip the cumulative-return PNG")
    args = parser.parse_args()
    for c in args.configs:
        print(f"\n=== {c} ===")
        run_config(c, args.out, plot=not args.no_plot)


if __name__ == "__main__":
    main()
