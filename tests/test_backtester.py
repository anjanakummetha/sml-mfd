"""Backtester self-tests. Run from the repository root after changing src/backtester.py:

    python tests/test_backtester.py        (or: pytest tests/test_backtester.py)

Each check returns (name, status, detail) with status PASS / FAIL / SKIP. A check is skipped when its
input file isn't in data/ (data/ is git-ignored; see README). The dashboard's Diagnostics tab runs the same checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src import backtester as bt  # noqa: E402

REG = json.loads((REPO_ROOT / "config/backtest/datasets.json").read_text())
MFD = "MFD · 60-month window (2005–2015)"
LAST_YEAR = "Last year · predicted returns (2020–2025)"


def _path(name):
    return REPO_ROOT / REG["datasets"][name]["path"]


def _load(name):
    d = REG["datasets"][name]
    return bt.load_panel(REPO_ROOT / d["path"], ret_col=d["ret_col"], me_col=d.get("me_col"), verbose=False)


def _skip(name, dataset):
    return [(name, "SKIP", f"{REG['datasets'][dataset]['path']} not found")]


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
    if not _path(LAST_YEAR).exists():
        return _skip("Reproduce last year's Task 7", LAST_YEAR)
    out = []
    panel = _load(LAST_YEAR)
    for preset, tgt in REG["reproduction_targets"].items():
        if preset.startswith("_"):
            continue
        p = {k: v for k, v in REG["presets"][preset].items() if k != "dataset"}
        res = bt.run_strategy(panel, bt.Strategy(name=preset, **p))
        m = bt.perf_metrics(res.monthly.strategy_ret)
        ok = abs(m["mean_monthly"] - tgt["mean_monthly"]) < 5e-4 and abs(m["t_stat"] - tgt["t_stat"]) < 0.03
        out.append((f"Reproduce: {preset.split('· ')[-1]}", "PASS" if ok else "FAIL",
                    f"{m['mean_monthly']:.2%}/mo, t = {m['t_stat']:.2f} "
                    f"(last year {tgt['mean_monthly']:.2%}, t = {tgt['t_stat']:.2f})"))
    return out


def check_holding_vs_brute_force(Ks=(1, 2, 3, 6)):
    if not _path(MFD).exists():
        return _skip("Holding periods vs brute force", MFD)
    out = []
    panel = _load(MFD)
    for K in Ks:
        res = bt.run_strategy(panel, bt.Strategy(name="t", signal={"mfd": 1}, tail_pct=10, holding_months=K))
        b = brute_force_ls(panel, "mfd", K)
        same_months = res.monthly.index.equals(pd.DatetimeIndex(b.index))
        diff = (res.monthly.ls_ret - b.reindex(res.monthly.index)).abs().max()
        out.append((f"{K}-month hold vs brute force", "PASS" if same_months and diff < 1e-12 else "FAIL",
                    f"{len(res.monthly)} months, max difference {diff:.1e}"))
    return out


def check_ramp_up_modes():
    if not _path(MFD).exists():
        return _skip("Ramp-up: cash vs drop", MFD)
    panel = _load(MFD)
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
    if not _path(MFD).exists():
        return _skip("Built-in checks", MFD)
    panel = _load(MFD)
    out = []
    for K, w in ((1, "equal"), (12, "value")):
        res = bt.run_strategy(panel, bt.Strategy(name="s", signal={"mfd": 1}, tail_pct=10, holding_months=K,
                                                 weighting=w))
        failed = [c["check"] for c in bt.sanity_checks(res) if not c["ok"]]
        out.append((f"Built-in checks ({K}-month hold, {w})", "FAIL" if failed else "PASS",
                    f"failed: {failed}" if failed else "all pass"))
    return out


ALL_CHECKS = [check_reproduce_last_year, check_holding_vs_brute_force, check_ramp_up_modes,
              check_built_in_sanity_checks]


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
        print(f"{status:<5} {name:<42} {detail}")
    n = {s: sum(r[1] == s for r in res) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    sys.exit(1 if n["FAIL"] else 0)
