"""Backtester self-tests. Run from the repository root after changing src/backtester.py:

    python tests/test_backtester.py        (or: pytest tests/test_backtester.py)

Each check returns (name, status, detail) with status PASS / FAIL / SKIP. A check is skipped when its
input file isn't in data/ (data/ is git-ignored; see README). The dashboard's Diagnostics tab runs the same checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src import backtester as bt  # noqa: E402

MFD_FILE = REPO_ROOT / "data" / "mfd_60.csv"
LAST_YEAR_FILE = REPO_ROOT / "data" / "lastyear_bbl_predictions.csv"
# Last year's Task 7: top 10% minus bottom 10% by predicted return, equal-weighted, 70 months, pd.qcut ties.
REPRODUCTION_TARGETS = {"accounting_prediction": (0.015213, 1.9927), "market_prediction": (0.028384, 2.8897)}
# Strategy types each known file must offer (a file not in data/ is skipped)
EXPECTED_TYPES = {
    MFD_FILE: list(bt.STRATEGY_TYPES),
    LAST_YEAR_FILE: ["cross_track_mfd", "accounting_prediction", "market_prediction"],
}


def _skip(name, path):
    return [(name, "SKIP", f"{Path(path).relative_to(REPO_ROOT)} not found")]


def _load(path):
    return bt.load_dataset(path, verbose=False)


def brute_force_ls(panel, signal, K, tail=10):
    """Independent, loop-based K-month overlapping long-short (equal-weighted, low minus high)."""
    p = panel.dropna(subset=[signal]).copy()
    p["pct"] = p.groupby("eom")[signal].rank(pct=True) * 100
    months = sorted(p.eom.unique())
    ret = {(i, e): r for i, e, r in zip(p.id, p.eom, p.ret) if pd.notna(r)}
    cohorts = {e: (list(g.id[g.pct <= tail]), list(g.id[g.pct > 100 - tail])) for e, g in p.groupby("eom")}
    out = {}
    for j, e in enumerate(months):              # returns earned in the month after `e`
        live = []
        for h in range(K):                      # cohort formed h months before `e`
            if j - h < 0:
                continue
            L, S = cohorts[months[j - h]]
            lr = [ret[(i, e)] for i in L if (i, e) in ret]
            sr = [ret[(i, e)] for i in S if (i, e) in ret]
            if lr and sr:
                live.append(np.mean(lr) - np.mean(sr))
        if len(live) == K:
            out[e + pd.offsets.MonthEnd(1)] = np.mean(live)
    return pd.Series(out)


def check_reproduce_last_year():
    if not LAST_YEAR_FILE.exists():
        return _skip("Reproduce last year's Task 7", LAST_YEAR_FILE)
    out = []
    panel, info = _load(LAST_YEAR_FILE)
    for t, (mean, tstat) in REPRODUCTION_TARGETS.items():
        signal, direction = bt.resolve_type(t, info["roles"])
        res = bt.run_strategy(panel, bt.Strategy(name=t, signal=signal, strategy_type=t, direction=direction,
                                                 tail_pct=10, tie_method="min"))
        m = bt.perf_metrics(res.monthly.strategy_ret)
        ok = abs(m["mean_monthly"] - mean) < 5e-4 and abs(m["t_stat"] - tstat) < 0.03
        out.append((f"Reproduce: {bt.STRATEGY_TYPES[t]['label'].split(' (')[0].lower()}", "PASS" if ok else "FAIL",
                    f"{m['mean_monthly']:.2%}/mo, t = {m['t_stat']:.2f} (last year {mean:.2%}, t = {tstat:.2f})"))
    return out


def check_holding_vs_brute_force(Ks=(1, 2, 3, 6)):
    if not MFD_FILE.exists():
        return _skip("Holding periods vs brute force", MFD_FILE)
    out = []
    panel, _ = _load(MFD_FILE)
    for K in Ks:
        res = bt.run_strategy(panel, bt.Strategy(name="t", signal={"mfd": 1}, tail_pct=10, holding_months=K))
        b = brute_force_ls(panel, "mfd", K)
        same_months = res.monthly.index.equals(pd.DatetimeIndex(b.index))
        diff = (res.monthly.ls_ret - b.reindex(res.monthly.index)).abs().max()
        out.append((f"{K}-month hold vs brute force", "PASS" if same_months and diff < 1e-12 else "FAIL",
                    f"{len(res.monthly)} months, max difference {diff:.1e}"))
    return out


def check_ramp_up_modes():
    if not MFD_FILE.exists():
        return _skip("Ramp-up: cash vs drop", MFD_FILE)
    panel, _ = _load(MFD_FILE)
    kw = dict(name="r", signal={"mfd": 1}, tail_pct=10, holding_months=3)
    cash = bt.run_strategy(panel, bt.Strategy(**kw, ramp_up="cash")).monthly
    drop = bt.run_strategy(panel, bt.Strategy(**kw, ramp_up="drop")).monthly
    same = np.allclose(cash.ls_ret.loc[drop.index], drop.ls_ret)
    one_cohort = bt.run_strategy(panel, bt.Strategy(**{**kw, "holding_months": 1})).monthly.ls_ret.iloc[0]
    ok = same and len(cash) == len(drop) + 2 and np.isclose(cash.ls_ret.iloc[0], one_cohort / 3)
    return [("Ramp-up: cash vs drop", "PASS" if ok else "FAIL",
             f"identical once 3 cohorts are live; cash keeps {len(cash) - len(drop)} extra months, "
             f"first month = 1/3 of a single cohort")]


def check_built_in_sanity_checks():
    if not MFD_FILE.exists():
        return _skip("Built-in checks", MFD_FILE)
    panel, _ = _load(MFD_FILE)
    out = []
    for K, w in ((1, "equal"), (12, "value")):
        res = bt.run_strategy(panel, bt.Strategy(name="s", signal={"mfd": 1}, tail_pct=10, holding_months=K,
                                                 weighting=w))
        failed = [c["check"] for c in bt.sanity_checks(res) if not c["ok"]]
        out.append((f"Built-in checks ({K}-month hold, {w})", "FAIL" if failed else "PASS",
                    f"failed: {failed}" if failed else "all pass"))
    return out


def check_pooled_mfd_formula():
    """The pooled MFD derived from per-track columns equals the SD of all investors' forecasts."""
    rng = np.random.default_rng(7)
    acct = rng.normal(0.006, 0.002, size=(500, 50)) + rng.normal(0, 0.003, size=(500, 1))
    mkt = rng.normal(0.004, 0.003, size=(500, 50)) + rng.normal(0, 0.003, size=(500, 1))
    direct = np.concatenate([acct, mkt], axis=1).std(axis=1, ddof=0)
    df = pd.DataFrame({"am": acct.mean(1), "mm": mkt.mean(1), "asd": acct.std(1, ddof=0), "msd": mkt.std(1, ddof=0)})
    bt.add_pooled_mfd(df, {"accounting_forecast": "am", "market_forecast": "mm", "accounting_sd": "asd",
                           "market_sd": "msd"})
    diff = float(np.abs(df["pooled_mfd"].to_numpy() - direct).max())
    return [("Pooled MFD formula", "PASS" if diff < 1e-12 else "FAIL",
             f"matches the SD of all 100 simulated forecasts (max difference {diff:.1e})")]


def check_column_detection():
    """Files with other column names are recognized through config/backtest/columns.json."""
    renamed = bt.detect_columns(["permno", "date", "ret_exc_lead1m", "prediction_market", "prediction_accounting"])
    no_return = bt.detect_columns(["id", "eom", "mfd"])
    ok = (renamed is not None and renamed["columns"]["id"] == "permno" and renamed["columns"]["date"] == "date"
          and bt.available_types(renamed["roles"]) == ["accounting_prediction", "market_prediction"]
          and no_return is None)
    return [("Column detection", "PASS" if ok else "FAIL",
             "alternative column names map to roles; a file without a return column is rejected")]


def check_strategy_types():
    """Every strategy type runs on every data file in data/ that supports it, with all built-in checks."""
    out = []
    found = bt.discover_datasets()
    for path, expected in EXPECTED_TYPES.items():
        name = bt.dataset_name(path)
        if not path.exists():
            out.extend(_skip(f"Strategy types offered for {name}", path))
            continue
        got = found.get(name, {}).get("types", [])
        out.append((f"Strategy types offered for {name}", "PASS" if got == expected else "FAIL",
                    f"{len(got)} types" + ("" if got == expected else f": {got}, expected {expected}")))
    for name, d in found.items():
        panel, info = _load(d["path"])
        failed = []
        for t in info["types"]:
            signal, direction = bt.resolve_type(t, info["roles"])
            res = bt.run_strategy(panel, bt.Strategy(name=t, signal=signal, strategy_type=t, direction=direction))
            failed += [f"{t}: {c['check']}" for c in bt.sanity_checks(res) if not c["ok"]]
        out.append((f"All strategy types run on {name}", "FAIL" if failed else "PASS",
                    f"failed: {failed}" if failed else f"{len(info['types'])} types, all built-in checks pass"))
    return out


ALL_CHECKS = [check_reproduce_last_year, check_holding_vs_brute_force, check_ramp_up_modes,
              check_built_in_sanity_checks, check_pooled_mfd_formula, check_column_detection,
              check_strategy_types]


def run_all():
    results = []
    for c in ALL_CHECKS:
        results.extend(c())
    return results


def test_backtester():
    """pytest entry point: every check passes or is skipped for missing data."""
    failed = [r for r in run_all() if r[1] == "FAIL"]
    assert not failed, failed


if __name__ == "__main__":
    res = run_all()
    for name, status, detail in res:
        print(f"{status:<5} {name:<44} {detail}")
    n = {s: sum(r[1] == s for r in res) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    sys.exit(1 if n["FAIL"] else 0)
