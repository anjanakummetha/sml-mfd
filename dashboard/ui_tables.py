"""Research tables for the UI (and for anything else that wants them). Returns in DECIMALS unless noted;
formatting to % happens in the app."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtester as bt

FACTORS_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def alphas(r: pd.Series, factors: pd.DataFrame | None, lags: int) -> dict:
    """{model: (alpha, t)} with Newey-West t-stats; empty if no factor data overlaps."""
    if factors is None:
        return {}
    fr = bt.factor_regressions(r, factors, lags)
    return {row.model: (row.alpha, row.alpha_t_nw) for row in fr.itertuples()}


def stats_row(r: pd.Series, lags: int) -> dict:
    m = bt.perf_metrics(r, lags)
    return {"Mean": m["mean_monthly"], "t (NW)": m["t_stat_nw"], "SD": m["ann_vol"] / np.sqrt(12),
            "Sharpe": m["sharpe"], "Sortino": m["sortino"], "Max DD": m["max_drawdown"],
            "Skew": m["skew"], "Ex. kurt": m["excess_kurtosis"], "Min": m["worst_month"], "Max": m["best_month"],
            "Months": m["n_months"]}


def summary_table(monthly: pd.DataFrame, s: bt.Strategy, factors, lags: int) -> pd.DataFrame:
    rows = []
    if "long_ret" in monthly:
        rows.append((f"Long  (P{s.long_pct[0]:g}–{s.long_pct[1]:g})", monthly["long_ret"]))
    if "short_ret" in monthly:
        rows.append((f"Short (P{s.short_pct[0]:g}–{s.short_pct[1]:g})", monthly["short_ret"]))
    if "ls_ret" in monthly:
        rows.append(("Long − short", monthly["ls_ret"]))
    out = []
    for name, r in rows:
        a = alphas(r, factors, lags)
        out.append({"Portfolio": name, **stats_row(r, lags),
                    "FF6 α": a.get("FF6", (np.nan,))[0], "α t": a.get("FF6", (np.nan, np.nan))[1]})
    return pd.DataFrame(out)


def decile_table(buckets: pd.DataFrame, chars: pd.DataFrame, factors, lags: int) -> pd.DataFrame:
    """One row per bucket plus the bottom-minus-top spread: mean, NW t, SD, Sharpe, CAPM and FF6 alphas,
    average stocks and formation characteristics."""
    p = buckets.pivot(index="month", columns="bucket", values="ret")
    n = buckets.groupby("bucket")["n"].mean()
    series = [(str(b), p[b]) for b in p.columns] + [(f"1 − {p.columns[-1]}", p[p.columns[0]] - p[p.columns[-1]])]
    rows = []
    for name, r in series:
        m = bt.perf_metrics(r, lags)
        a = alphas(r, factors, lags)
        row = {"Bucket": name, "Mean": m["mean_monthly"], "t (NW)": m["t_stat_nw"],
               "SD": m["ann_vol"] / np.sqrt(12), "Sharpe": m["sharpe"],
               "CAPM α": a.get("CAPM", (np.nan,))[0], "CAPM t": a.get("CAPM", (np.nan, np.nan))[1],
               "FF6 α": a.get("FF6", (np.nan,))[0], "FF6 t": a.get("FF6", (np.nan, np.nan))[1]}
        if name.isdigit():
            b = int(name)
            row["Avg. N"] = n.get(b, np.nan)
            for c in chars.columns:
                row[c] = chars.loc[b, c] if b in chars.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def characteristics(fp: pd.DataFrame, s: bt.Strategy) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series averages of monthly cross-sectional characteristics, by bucket and by leg."""
    def summarize(groups):
        out = {}
        if "me" in fp:
            out["Median ME ($M)"] = groups["me"].median().groupby(level=-1).mean()
        for c in s.signal:
            out[f"Mean {c} (%)"] = groups[c].mean().groupby(level=-1).mean() * 100
        return pd.DataFrame(out)

    by_bucket = summarize(fp.groupby(["fm", "bucket"]))
    legs = []
    for side, rng in (("Long", s.long_pct), ("Short", s.short_pct)):
        if rng is None:
            continue
        lo, hi = rng
        mask = ((fp.pct > lo) & (fp.pct <= hi)) if lo > 0 else (fp.pct <= hi)
        sub = fp[mask].assign(leg=side)
        legs.append(summarize(sub.groupby(["fm", "leg"])))
    by_leg = pd.concat(legs) if legs else pd.DataFrame()
    return by_bucket, by_leg


def factor_table(r: pd.Series, factors: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Paper-style regression table: coefficient row, then (t-stat) row. α in % per month."""
    fr = bt.factor_regressions(r, factors, lags).set_index("model")
    models = list(fr.index)
    rows = []

    def two(label, coef, tval, fmt):
        rows.append([label] + [("" if pd.isna(c) else fmt.format(c)) for c in coef])
        rows.append([""] + [("" if pd.isna(v) else f"({v:.2f})") for v in tval])

    two("α (% / month)", [fr.loc[m, "alpha"] * 100 for m in models], [fr.loc[m, "alpha_t_nw"] for m in models],
        "{:.2f}")
    for f in FACTORS_ORDER:
        two(f, [fr.loc[m].get(f"beta_{f}", np.nan) for m in models],
            [fr.loc[m].get(f"t_{f}", np.nan) for m in models], "{:.2f}")
    rows.append(["R²"] + [f"{fr.loc[m, 'r2']:.2f}" for m in models])
    rows.append(["Months"] + [f"{int(fr.loc[m, 'n'])}" for m in models])
    return pd.DataFrame(rows, columns=["", *models])


def subperiod_table(r: pd.Series, factors, lags: int) -> pd.DataFrame:
    r = r.dropna()
    idx = r.index
    half = len(idx) // 2
    periods = [("Full sample", r),
               (f"{idx[0]:%Y-%m} to {idx[half - 1]:%Y-%m}", r.iloc[:half]),
               (f"{idx[half]:%Y-%m} to {idx[-1]:%Y-%m}", r.iloc[half:])]
    crisis = (idx >= pd.Timestamp("2008-01-01")) & (idx <= pd.Timestamp("2009-12-31"))
    if crisis.any() and (~crisis).sum() >= 24:
        periods.append(("Excluding 2008–2009", r[~crisis]))
    rows = []
    for name, x in periods:
        if len(x) < 12:
            continue
        m = bt.perf_metrics(x, lags)
        a = alphas(x, factors, lags).get("FF6", (np.nan, np.nan))
        rows.append({"Period": name, "Months": m["n_months"], "Mean": m["mean_monthly"], "t (NW)": m["t_stat_nw"],
                     "Sharpe": m["sharpe"], "Max DD": m["max_drawdown"], "FF6 α": a[0], "α t": a[1]})
    return pd.DataFrame(rows)


def ic_table(ic: pd.Series) -> pd.DataFrame:
    ic = ic.dropna()
    return pd.DataFrame([{"Mean IC": ic.mean(), "SD": ic.std(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic))),
                          "IC > 0": (ic > 0).mean(), "Months": len(ic)}])
