"""Research dashboard for the Task 7 disagreement backtester. From the repository root:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DASH = Path(__file__).resolve().parent
REPO_ROOT = DASH.parent
sys.path[:0] = [str(REPO_ROOT), str(DASH), str(REPO_ROOT / "tests")]
from src import backtester as bt   # noqa: E402
import ui_charts as uc             # noqa: E402
import ui_tables as ut             # noqa: E402

st.set_page_config(page_title="Disagreement Portfolio Sorts", layout="wide")

REG = json.loads((REPO_ROOT / "config/backtest/datasets.json").read_text())
# only datasets whose file is present locally (data/ is git-ignored)
DATASETS = {k: d for k, d in REG["datasets"].items() if (REPO_ROOT / d["path"]).exists()}
PRESETS = {k: p for k, p in REG["presets"].items() if p["dataset"] in DATASETS}
if not DATASETS:
    st.error("No backtest data found. Copy mfd_60.csv from $SML_OUTPUT_DIR on NVwulf into data/ "
             "(see config/backtest/datasets.json and the README).")
    st.stop()
MODE = "dark" if (getattr(st.context, "theme", None) and st.context.theme.type == "dark") else "light"
T = uc.TOKENS[MODE]
MAX_RUNS = 8
EXPORT_DIR = REPO_ROOT / "results" / "backtest" / "ui_runs"

# --------------------------------------------------------------------------------------
# Data + engine (cached)
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data…")
def load_data(key: str) -> pd.DataFrame:
    d = DATASETS[key]
    return bt.load_panel(REPO_ROOT / d["path"], ret_col=d["ret_col"], me_col=d.get("me_col"), verbose=False)


@st.cache_data(show_spinner=False)
def load_factors() -> pd.DataFrame:
    return bt.load_ff_factors(REPO_ROOT / "config/factors")


def signal_columns(key: str) -> list[str]:
    panel, d = load_data(key), DATASETS[key]
    skip = {"id", "eom", "fm", "ret", "me", *d.get("not_signals", [])}
    return [c for c in panel.columns if c not in skip and pd.api.types.is_numeric_dtype(panel[c])]


def label(key: str, col: str) -> str:
    return DATASETS[key].get("labels", {}).get(col, col)


if not PRESETS:                              # data present but no preset for it: start from its first score
    _ds = next(iter(DATASETS))
    PRESETS = {"Default": {"dataset": _ds, "signal": {signal_columns(_ds)[0]: 1}}}


@st.cache_data(show_spinner="Running backtest…", max_entries=64)
def run(key: str, strat_json: str):
    s = bt.Strategy(**json.loads(strat_json))
    panel, F = load_data(key), load_factors()
    res = bt.run_strategy(panel, s, REPO_ROOT)
    summ, fr = bt.summarize(res, F)
    chars_bucket, chars_leg = ut.characteristics(bt.formation_panel(panel, s, REPO_ROOT), s)
    tables = {
        "summary": ut.summary_table(res.monthly, s, F, s.nw_lags),
        "deciles": ut.decile_table(res.buckets, chars_bucket, F, s.nw_lags),
        "chars_leg": chars_leg,
        "subperiods": ut.subperiod_table(res.monthly["strategy_ret"], F, s.nw_lags),
        "ic": ut.ic_table(res.ic),
    }
    return res, summ, fr, bt.sanity_checks(res), tables


@st.cache_data(show_spinner=False, max_entries=64)
def factor_table(key: str, strat_json: str, series: str):
    res = run(key, strat_json)[0]
    return ut.factor_table(res.monthly[series], load_factors(), res.strategy.nw_lags)


@st.cache_data(show_spinner="Running engine tests (about a minute)…")
def engine_tests():
    import test_backtester
    return test_backtester.run_all()

# --------------------------------------------------------------------------------------
# Settings state. st.session_state.cfg is the source of truth; widgets take their starting value from it
# and a key with a version number, which is bumped on a preset or dataset change to recreate them.
# --------------------------------------------------------------------------------------

DEFAULTS = dict(weights={}, standardize="rank", long_rng=(0, 10), short_rng=(90, 100), use_short=True,
                weighting="equal", hold=1, ramp="drop", period=None, drop_size=0, tie="average", nbuckets=10, nw=6)


def months_of(key):
    return [m.strftime("%Y-%m") for m in sorted(load_data(key).eom.unique())]


def apply_preset():
    p = PRESETS[st.session_state.preset]
    ds, ms = p["dataset"], months_of(p["dataset"])
    st.session_state.dataset = ds
    st.session_state.cfg = {**DEFAULTS, "weights": dict(p["signal"]),
                            "long_rng": tuple(p.get("long_pct", (0, 10))),
                            "short_rng": tuple(p.get("short_pct") or (90, 100)),
                            "use_short": p.get("short_pct", [90, 100]) is not None,
                            "tie": p.get("tie_method", "average"), "period": (ms[0], ms[-1])}
    st.session_state.v += 1
    st.session_state.pending_run = True


def on_dataset_change():
    ds, ms = st.session_state.dataset, months_of(st.session_state.dataset)
    cfg = st.session_state.cfg
    cfg.update(weights={signal_columns(ds)[0]: 1.0}, period=(ms[0], ms[-1]))
    if "me" not in load_data(ds):
        cfg.update(weighting="equal", drop_size=0)
    st.session_state.v += 1


if "cfg" not in st.session_state:
    st.session_state.update(v=0, runs=[], preset=next(iter(PRESETS)), active=None, pending_run=False)
    apply_preset()


def build_strategy(cfg: dict) -> dict:
    start, end = cfg["period"]
    return asdict(bt.Strategy(
        name="ui", signal={k: float(v) for k, v in cfg["weights"].items() if v},
        standardize=cfg["standardize"], long_pct=list(cfg["long_rng"]),
        short_pct=list(cfg["short_rng"]) if cfg["use_short"] else None,
        weighting=cfg["weighting"], holding_months=cfg["hold"], ramp_up=cfg["ramp"], n_buckets=cfg["nbuckets"],
        start=start, end=end, drop_size_pct=float(cfg["drop_size"]), nw_lags=cfg["nw"], tie_method=cfg["tie"]))

# --------------------------------------------------------------------------------------
# Sidebar: specification
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Specification")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.selectbox("Dataset", list(DATASETS), key="dataset", on_change=on_dataset_change,
                 help="Listed in config/backtest/datasets.json; only files present in data/ are shown.")
    ds, cfg, v = st.session_state.dataset, st.session_state.cfg, st.session_state.v
    has_me = "me" in load_data(ds)
    ms = months_of(ds)

    def k(name):
        return f"{name}_{v}"

    with st.form("settings", border=False):
        st.markdown("**Sorting variable**")
        cols = signal_columns(ds)
        editor = pd.DataFrame({"signal": cols, "description": [label(ds, c) for c in cols],
                               "weight": [cfg["weights"].get(c, 0.0) for c in cols]})
        edited = st.data_editor(
            editor, key=k(f"weights_{ds}"), hide_index=True, width="stretch", disabled=["signal", "description"],
            column_config={"signal": None,
                           "description": st.column_config.TextColumn("Score", width=165),
                           "weight": st.column_config.NumberColumn("Weight", step=0.25, format="%.2f", width=70)})
        st.caption("Each score with a non-zero weight is standardized within the month, then summed. "
                   "SD = cross-investor standard deviation of forecasts.")
        standardize = st.segmented_control(
            "Standardization", ["rank", "zscore", "raw"], default=cfg["standardize"], key=k("std"),
            help="rank: cross-sectional percentile rank. zscore: (x − mean) / sd. raw: unscaled.")

        st.markdown("**Portfolio formation**")
        long_rng = st.slider("Long: score percentiles", 0, 100, value=tuple(cfg["long_rng"]), key=k("long"),
                             help="0 = lowest score. The paper's trade is long the lowest-disagreement decile.")
        use_short = st.toggle("Short leg", value=cfg["use_short"], key=k("use_short"))
        short_rng = st.slider("Short: score percentiles", 0, 100, value=tuple(cfg["short_rng"]), key=k("short"))
        weighting = st.segmented_control(
            "Weighting", ["equal", "value"] if has_me else ["equal"], default=cfg["weighting"], key=k("wt"),
            help="value: market cap at formation." + ("" if has_me else " Unavailable: no market cap in this dataset."))
        hold = st.slider("Holding period K (months)", 1, 12, value=cfg["hold"], key=k("hold"),
                         help="Overlapping portfolios: each month a new cohort receives 1/K of capital and is held "
                              "K months (Jegadeesh-Titman). K = 1 is monthly rebalancing.")
        ramp = st.segmented_control(
            "First K−1 months", ["drop", "cash"], default=cfg["ramp"], key=k("ramp"),
            help="drop: start once all K cohorts are live. cash: keep them, uninvested share earns 0.")

        st.markdown("**Sample**")
        period = st.select_slider("Formation months", options=ms, value=tuple(cfg["period"]), key=k(f"period_{ds}"))
        drop_size = st.slider("Exclude smallest stocks (% by ME)", 0, 50, value=cfg["drop_size"], step=5,
                              key=k("drop"), disabled=not has_me)

        st.markdown("**Inference**")
        nbuckets = st.slider("Number of buckets", 3, 20, value=cfg["nbuckets"], key=k("nb"))
        nw = st.slider("Newey-West lags", 0, 12, value=cfg["nw"], key=k("nw"))
        tie = st.segmented_control("Ties", ["average", "min"], default=cfg["tie"], key=k("tie"),
                                   help="min reproduces pd.qcut (last year's code): ties go to the lower bucket.")
        submitted = st.form_submit_button("Run", type="primary", width="stretch")

    if submitted:
        st.session_state.cfg = {
            "weights": dict(zip(edited.signal, edited.weight.fillna(0))),
            "standardize": standardize or cfg["standardize"], "long_rng": tuple(long_rng),
            "short_rng": tuple(short_rng), "use_short": use_short, "weighting": weighting or cfg["weighting"],
            "hold": hold, "ramp": ramp or cfg["ramp"], "period": tuple(period), "drop_size": drop_size,
            "nbuckets": nbuckets, "nw": nw, "tie": tie or cfg["tie"]}
        st.session_state.pending_run = True

# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------

if st.session_state.pending_run:
    st.session_state.pending_run = False
    cfg = st.session_state.cfg
    if not any(cfg["weights"].values()):
        st.error("Give at least one score a non-zero weight.")
    else:
        if cfg["use_short"] and max(cfg["long_rng"][0], cfg["short_rng"][0]) < min(cfg["long_rng"][1],
                                                                                 cfg["short_rng"][1]):
            st.warning("The long and short percentile ranges overlap.")
        st.session_state.active = (st.session_state.dataset, json.dumps(build_strategy(cfg), sort_keys=True))

if not st.session_state.active:
    st.stop()

ACTIVE_DS, ACTIVE_JSON = st.session_state.active
try:
    res, summ, fr, checks, tables = run(ACTIVE_DS, ACTIVE_JSON)
except Exception as e:                       # show engine errors instead of a blank page
    st.error(f"The backtest failed: {e}")
    st.stop()
S, mo = res.strategy, res.monthly
F = load_factors()

# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------

PCT = lambda x: "" if pd.isna(x) else f"{x * 100:.2f}"          # noqa: E731
NUM = lambda x: "" if pd.isna(x) else f"{x:.2f}"                 # noqa: E731
INT = lambda x: "" if pd.isna(x) else f"{x:,.0f}"                # noqa: E731


def show_table(df: pd.DataFrame, fmt: dict, height=None):
    st.dataframe(df.style.format(fmt, na_rep=""), hide_index=True, width="stretch",
                 **({"height": height} if height else {}))


def show(fig):
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


def spec_lines(s: bt.Strategy, key: str) -> list[str]:
    sig = " + ".join(c if w == 1 else f"{w:g}·{c}" for c, w in s.signal.items())
    legs = f"long P{s.long_pct[0]:g}–{s.long_pct[1]:g}" + (f", short P{s.short_pct[0]:g}–{s.short_pct[1]:g}"
                                                         if s.short_pct else " (long-only)")
    hold = f"K = {s.holding_months}" + (f" ({s.ramp_up} first K−1 months)" if s.holding_months > 1 else "")
    sample = f"formation {s.start} to {s.end}" + (f", smallest {s.drop_size_pct:g}% by ME excluded"
                                                  if s.drop_size_pct else "")
    return [f"{key}",
            f"Sort on {sig} ({s.standardize}) · {legs} · {s.weighting}-weighted · {hold}",
            f"{sample} · {len(mo)} return months, {mo.index[0]:%Y-%m} to {mo.index[-1]:%Y-%m} · "
            f"Newey-West lags {s.nw_lags}"]


def short_label(s: bt.Strategy) -> str:
    sig = "+".join(s.signal) if len(s.signal) <= 2 else f"{len(s.signal)} scores"
    legs = f"P{s.long_pct[0]:g}–{s.long_pct[1]:g}" + (f"/P{s.short_pct[0]:g}–{s.short_pct[1]:g}" if s.short_pct
                                                      else " long-only")
    return f"{sig} · {legs} · {'EW' if s.weighting == 'equal' else 'VW'} · K={s.holding_months}"


RET_NOTE = "Returns are monthly excess returns in percent. t-statistics use Newey-West standard errors " \
           f"({S.nw_lags} lags). Sharpe and Sortino ratios are annualized. FF6 = Fama-French five factors + momentum."

# --------------------------------------------------------------------------------------
# Header + summary table
# --------------------------------------------------------------------------------------

st.markdown("### Disagreement portfolio sorts")
st.caption("  \n".join(spec_lines(S, ACTIVE_DS)))
if st.session_state.dataset != ACTIVE_DS:
    st.info("The dataset selector has changed; press Run to load it.")

failed = [c for c in checks if not c["ok"]]
if failed:
    st.error(f"{len(failed)} of {len(checks)} engine checks failed: "
             + "; ".join(f"{c['check']} ({c['detail']})" for c in failed) + ". See Diagnostics.")
else:
    st.caption(f"Engine checks: {len(checks)} of {len(checks)} passed (see Diagnostics).")

st.markdown("**Summary statistics**")
show_table(tables["summary"], {"Mean": PCT, "t (NW)": NUM, "SD": PCT, "Sharpe": NUM, "Sortino": NUM,
                               "Max DD": PCT, "Skew": NUM, "Ex. kurt": NUM, "Min": PCT, "Max": PCT,
                               "Months": INT, "FF6 α": PCT, "α t": NUM})
st.caption(RET_NOTE + " Short-leg statistics are for the stocks sold (the leg's own return).")

# --------------------------------------------------------------------------------------
# Exhibits
# --------------------------------------------------------------------------------------

runs = st.session_state.runs
tabs = st.tabs(["Portfolio sorts", "Performance", "Factor regressions", "Robustness", "Characteristics & turnover",
                f"Compare ({len(runs)})", "Diagnostics & export"])

with tabs[0]:
    d = tables["deciles"]
    st.markdown(f"**Returns of {S.n_buckets} portfolios sorted on the score**")
    fmt = {"Mean": PCT, "t (NW)": NUM, "SD": PCT, "Sharpe": NUM, "CAPM α": PCT, "CAPM t": NUM, "FF6 α": PCT,
           "FF6 t": NUM, "Avg. N": INT, "Median ME ($M)": INT}
    fmt |= {c: "{:.3f}" for c in d.columns if c.startswith("Mean ") and c.endswith("(%)")}
    show_table(d, fmt)
    st.caption(f"Bucket 1 holds the lowest scores. The last row is the bucket 1 minus bucket {S.n_buckets} spread. "
               f"{S.weighting.capitalize()}-weighted, K = {S.holding_months}. Median ME and score means are "
               "time-series averages of monthly cross-sectional values at formation. " + RET_NOTE)
    g = res.buckets.groupby("bucket")["ret"]
    a, b = st.columns(2)
    with a:
        show(uc.bucket_means(pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count())}), T))
    with b:
        show(uc.bucket_cumulative(res.buckets, T))

with tabs[1]:
    log = st.toggle("Log scale", key="log_perf")
    show(uc.cumulative(mo, T, log=log))
    a, b = st.columns(2)
    with a:
        show(uc.drawdown(mo, T))
    with b:
        show(uc.monthly_bars(mo, T))

with tabs[2]:
    choices = {"Long − short": "strategy_ret", "Long leg": "long_ret", "Short leg": "short_ret"}
    choices = {k_: c for k_, c in choices.items() if c in mo}
    pick = st.segmented_control("Dependent variable", list(choices), default=next(iter(choices)), key="fr_dep")
    ft = factor_table(ACTIVE_DS, ACTIVE_JSON, choices[pick or next(iter(choices))])
    st.dataframe(ft, hide_index=True, width="stretch", height=(len(ft) + 1) * 35 + 3)
    st.caption("Time-series regressions of monthly excess returns on factor returns of the same month. "
               f"Newey-West t-statistics ({S.nw_lags} lags) in parentheses. α in percent per month. "
               "Mkt-RF market excess return; SMB size; HML value; RMW profitability; CMA investment; "
               "Mom momentum (12–2).")

with tabs[3]:
    st.markdown("**Subperiods, long − short**")
    show_table(tables["subperiods"], {"Months": INT, "Mean": PCT, "t (NW)": NUM, "Sharpe": NUM, "Max DD": PCT,
                                      "FF6 α": PCT, "α t": NUM})
    st.caption("The sample is split at the middle return month. " + RET_NOTE)
    a, b = st.columns(2)
    with a:
        show(uc.annual_bars(mo["strategy_ret"], T))
    with b:
        win = st.segmented_control("Rolling window (months)", [12, 24, 36], default=24, key="rwin")
        show(uc.rolling_sharpe(mo, T, window=win or 24))
    st.markdown("**Rank information coefficient**")
    show_table(tables["ic"], {"Mean IC": "{:.3f}", "SD": "{:.3f}", "t": NUM, "IC > 0": "{:.0%}", "Months": INT})
    st.caption("Monthly Spearman correlation between the score at formation and the next-month excess return. "
               "A negative mean IC means low scores earn more.")
    show(uc.rank_ic(res.ic, T))

with tabs[4]:
    leg = tables["chars_leg"]
    rows = []
    for side in ("long", "short"):
        if f"turnover_{side}" not in mo:
            continue
        row = {"Leg": side.capitalize(), "Stocks held": mo[f"n_{side}"].mean(),
               "Turnover / month": mo[f"turnover_{side}"].mean()}
        if side.capitalize() in leg.index:
            row |= leg.loc[side.capitalize()].to_dict()
        rows.append(row)
    ct = pd.DataFrame(rows)
    fmt = {"Stocks held": INT, "Turnover / month": "{:.1%}", "Median ME ($M)": INT}
    fmt |= {c: "{:.3f}" for c in ct.columns if c.startswith("Mean ")}
    show_table(ct, fmt)
    st.caption("Stocks held counts distinct names across all live cohorts. Turnover = ½ Σ|w_t − w_{t−1}| of each "
               "leg's weights (each leg sums to 1), ignoring price drift. Characteristics are measured at formation.")
    a, b = st.columns(2)
    with a:
        show(uc.lines([(mo[f"n_{s}"], name, uc.LEG[s]) for s, name in (("long", "Long"), ("short", "Short"))
                       if f"n_{s}" in mo], "Stocks held per leg", T))
    with b:
        show(uc.lines([(mo[f"turnover_{s}"], name, uc.LEG[s]) for s, name in (("long", "Long"), ("short", "Short"))
                       if f"turnover_{s}" in mo], "Monthly turnover per leg", T, yfmt=".0%"))
    a, b = st.columns(2)
    with a:
        show(uc.lines([(res.universe["n_stocks"], "Universe", 0)], "Stocks in the universe (formation month)", T,
                      legend=False))
    with b:
        if S.holding_months > 1:
            show(uc.lines([(mo["n_cohorts"], "Cohorts", 0)], "Live cohorts per month", T, yfmt="d", legend=False))

with tabs[5]:
    a, b, _ = st.columns([1.2, 1, 3])
    with a:
        if st.button("Add current run", disabled=len(runs) >= MAX_RUNS, width="stretch"):
            slot = min(set(range(MAX_RUNS)) - {r["slot"] for r in runs})
            runs.append({"label": f"{len(runs) + 1}. {short_label(S)}", "slot": slot, "dataset": ACTIVE_DS,
                         "summary": summ, "returns": mo["strategy_ret"]})
            st.rerun()
    with b:
        if st.button("Clear", disabled=not runs, width="stretch"):
            st.session_state.runs = []
            st.rerun()
    if not runs:
        st.caption(f"Add up to {MAX_RUNS} runs to compare specifications.")
    else:
        keys = [("mean_monthly", "Mean", PCT), ("t_stat_nw", "t (NW)", NUM), ("sharpe", "Sharpe", NUM),
                ("max_drawdown", "Max DD", PCT), ("alpha_FF6", "FF6 α", PCT), ("alpha_FF6_t", "α t", NUM),
                ("avg_turnover_long", "Turnover L", PCT), ("avg_turnover_short", "Turnover S", PCT),
                ("n_months", "Months", INT)]
        tbl = pd.DataFrame([{"Specification": r["label"], "Dataset": r["dataset"],
                             **{lbl: r["summary"].get(k_, np.nan) for k_, lbl, _ in keys}} for r in runs])
        show_table(tbl, {lbl: f for _, lbl, f in keys})
        st.caption("Long − short statistics. " + RET_NOTE)
        log = st.toggle("Log scale", key="log_cmp")
        show(uc.compare(runs, T, log=log))


def save_run() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = EXPORT_DIR / f"{stamp}_{re.sub(r'[^A-Za-z0-9]+', '-', short_label(S)).strip('-')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({"dataset": ACTIVE_DS, **json.loads(ACTIVE_JSON)}, indent=2))
    tables["summary"].to_csv(out / "summary_statistics.csv", index=False)
    tables["deciles"].to_csv(out / "portfolio_sorts.csv", index=False)
    tables["subperiods"].to_csv(out / "subperiods.csv", index=False)
    mo.reset_index().to_csv(out / "monthly_returns.csv", index=False)
    res.buckets.to_csv(out / "bucket_returns_monthly.csv", index=False)
    if fr is not None:
        fr.to_csv(out / "factor_regressions.csv", index=False)
    pd.DataFrame([summ]).to_csv(out / "summary_all_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "engine_checks.csv", index=False)
    return out


with tabs[6]:
    st.markdown("**Engine checks for this run**")
    show_table(pd.DataFrame([{"Result": "PASS" if c["ok"] else "FAIL", "Check": c["check"], "Detail": c["detail"]}
                             for c in checks]), {})
    d = load_data(ACTIVE_DS)
    st.markdown("**Data coverage**")
    cov = pd.DataFrame([{"Rows": len(d), "Formation months": d.eom.nunique(),
                         "Rows without next-month return": int(d.ret.isna().sum()),
                         "Return months in this run": len(mo),
                         "Ramp-up months dropped": res.meta["warmup_months_dropped"]}])
    show_table(cov, {c: INT for c in cov.columns})

    st.markdown("**Engine tests**")
    st.caption("Reproduces last year's Task 7 numbers and checks the holding-period code against an independent "
               "loop implementation. Same as `python tests/test_backtester.py`.")
    if st.button("Run engine tests"):
        st.session_state.show_tests = True
    if st.session_state.get("show_tests"):
        tests = engine_tests()
        n = {k_: sum(t_[1] == k_ for t_ in tests) for k_ in ("PASS", "FAIL", "SKIP")}
        st.caption(f"{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped (skipped = data file not in data/).")
        show_table(pd.DataFrame([{"Result": status, "Test": name, "Detail": dt_}
                                 for name, status, dt_ in tests]), {})

    st.markdown("**Export**")
    a, _ = st.columns([1, 3])
    with a:
        if st.button("Save run to project folder", type="primary", width="stretch"):
            st.session_state.last_export = str(save_run())
    if st.session_state.get("last_export"):
        st.caption("Saved to:")
        st.code(st.session_state.last_export, language=None)
    st.caption("Writes the tables, monthly series, factor regressions, checks and settings to "
               "results/backtest/ui_runs/ (git-ignored). "
               "The download buttons below save through the browser; they don't work in the Claude app's preview "
               "pane, so open the app in Chrome or Safari to use them.")
    a, b, c = st.columns(3)
    a.download_button("monthly_returns.csv", mo.reset_index().to_csv(index=False), "monthly_returns.csv",
                      mime="text/csv", on_click="ignore", width="stretch")
    b.download_button("portfolio_sorts.csv", tables["deciles"].to_csv(index=False), "portfolio_sorts.csv",
                      mime="text/csv", on_click="ignore", width="stretch")
    c.download_button("settings.json", json.dumps({"dataset": ACTIVE_DS, **json.loads(ACTIVE_JSON)}, indent=2),
                      "settings.json", mime="application/json", on_click="ignore", width="stretch")

    st.markdown("**Settings** (paste into the `strategies` list of a config/backtest/*.json file for "
                "`python -m src.backtester`)")
    st.code(json.dumps(json.loads(ACTIVE_JSON), indent=2), language="json")
