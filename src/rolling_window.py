"""Build yearly forecast windows and Slurm job lists without using future data."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import pandas as pd

DATE_COL = "eom"
ID_COL = "id"
TARGET = "ret_exc_lead1m"
TARGET_HORIZON_MONTHS = 1  # ret_exc_lead1m is the return realised one month after eom
FORECAST_YEARS = list(range(2005, 2016))
TRAIN_MONTHS = 60

# Identifiers / raw returns that must never be used as predictors.
META_COLS = [
    "id", "eom", "ret_exc_lead1m", "obs_main", "exch_main", "common",
    "primary_sec", "gvkey", "date", "permno", "iid", "excntry", "me",
    "ret_exc", "ret", "gics", "sic", "ff49", "crsp_shrcd", "crsp_exchcd",
]


class LookAheadError(RuntimeError):
    """A training window could see information from its test period."""


# --------------------------------------------------------------------------- data

def load_panel(path, columns=None):
    """Read the stock-month panel from .csv or .parquet and parse `eom` as dates."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=columns)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def get_feature_cols(df, features_file=None):
    """Predictor columns: from a one-name-per-line file if given, else every non-meta column."""
    if features_file:
        lines = Path(features_file).read_text().splitlines()
        cols = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Features not in data: {missing[:10]}")
    else:
        cols = [c for c in df.columns if c not in META_COLS]

    leaky = [c for c in cols if "lead" in c.lower()]
    if leaky:
        raise LookAheadError(f"Forward-looking columns in the feature list: {leaky}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns (add them to META_COLS?): {non_numeric}")
    return cols


def prepare_panel(df, feature_cols):
    """Drop rows with no realised target, then zero-fill missing features."""
    out = df.dropna(subset=[TARGET]).copy()

    # Zero-fill means "median" only if features are rank-standardised around 0.
    medians = out[feature_cols].median()
    off_center = medians[medians.abs() > 0.25]
    if len(off_center):
        print(
            f"WARNING: {len(off_center)} features have a median far from 0 "
            f"(e.g. {list(off_center.index[:5])}). Zero-filling assumes centred features.",
            file=sys.stderr,
        )
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


# ------------------------------------------------------------------------ windows

def window_months(forecast_year, train_months=TRAIN_MONTHS):
    """(train_first, train_last, test_first, test_last) as monthly Periods."""
    test_first = pd.Period(f"{forecast_year}-01", freq="M")
    test_last = pd.Period(f"{forecast_year}-12", freq="M")
    return test_first - train_months, test_first - 1, test_first, test_last


def check_no_lookahead(train_dates, test_dates, forecast_year,
                       horizon=TARGET_HORIZON_MONTHS):
    """Raise if the training window overlaps or peeks into the test period."""
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise ValueError(f"{forecast_year}: empty training or test window")
    last_train = pd.Timestamp(train_dates.max()).to_period("M")
    first_test = pd.Timestamp(test_dates.min()).to_period("M")
    if last_train >= first_test:
        raise LookAheadError(
            f"{forecast_year}: last training month {last_train} is not before "
            f"first test month {first_test}"
        )
    # The label on the last training row is a return realised `horizon` months later;
    # it must already be known on the first forecast date.
    if last_train + horizon > first_test:
        raise LookAheadError(
            f"{forecast_year}: training label for {last_train} is realised after "
            f"the first forecast date {first_test}"
        )


def get_window(df, forecast_year, train_months=TRAIN_MONTHS):
    """Boolean masks (train, test) for one forecast year, built from `eom` dates."""
    months = df[DATE_COL].dt.to_period("M")
    train_first, train_last, test_first, test_last = window_months(forecast_year, train_months)
    train_mask = months.between(train_first, train_last)
    test_mask = months.between(test_first, test_last)
    check_no_lookahead(df.loc[train_mask, DATE_COL], df.loc[test_mask, DATE_COL], forecast_year)
    return train_mask, test_mask


def window_table(df, years=FORECAST_YEARS, train_months=TRAIN_MONTHS):
    """One row per forecast year with the train/test dates actually present in the data."""
    rows = []
    for year in years:
        train_mask, test_mask = get_window(df, year, train_months)
        train_dates = df.loc[train_mask, DATE_COL]
        test_dates = df.loc[test_mask, DATE_COL]
        rows.append({
            "forecast_year": year,
            "train_start": train_dates.min().date(),
            "train_end": train_dates.max().date(),
            "n_train_months": train_dates.nunique(),
            "test_start": test_dates.min().date(),
            "test_end": test_dates.max().date(),
            "n_test_months": test_dates.nunique(),
            "train_obs": int(train_mask.sum()),
            "test_obs": int(test_mask.sum()),
        })
    table = pd.DataFrame(rows)

    short_train = table.loc[table.n_train_months < train_months, "forecast_year"].tolist()
    if short_train:
        print(f"WARNING: fewer than {train_months} training months for {short_train}",
              file=sys.stderr)
    short_test = table.loc[table.n_test_months < 12, "forecast_year"].tolist()
    if short_test:
        print(f"NOTE: fewer than 12 test months for {short_test} "
              f"(usually the last year, where {TARGET} is not yet realised)", file=sys.stderr)
    return table


def print_window_table(table):
    cols = ["forecast_year", "train_start", "train_end", "n_train_months",
            "test_start", "test_end", "n_test_months"]
    print("=== Rolling windows (train strictly before test) ===")
    print(table[cols].to_string(index=False))
    print(f"Look-ahead check passed for all {len(table)} windows.\n")


# --------------------------------------------------------------------- SLURM jobs

def build_job_list(tracks, investors, years=FORECAST_YEARS):
    """Every (track, investor, year) combination, numbered from 0 for --array."""
    combos = itertools.product(tracks, investors, years)
    return pd.DataFrame(
        [{"task_id": i, "track": t, "investor": inv, "forecast_year": y}
         for i, (t, inv, y) in enumerate(combos)]
    )


def job_for_task(jobs_csv, task_id=None):
    """Look up this array task's (track, investor, year); defaults to SLURM_ARRAY_TASK_ID."""
    if task_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise RuntimeError("Not running inside a SLURM array job; pass task_id explicitly")
        task_id = int(env)
    jobs = pd.read_csv(jobs_csv)
    row = jobs.loc[jobs.task_id == task_id]
    if len(row) != 1:
        raise KeyError(f"task_id {task_id} not found in {jobs_csv}")
    return row.iloc[0].to_dict()


def slurm_cpus(default=-1):
    """Cores SLURM gave this task (so n_jobs doesn't grab the whole node)."""
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(value) if value else default


# ----------------------------------------------------------------- practice model

def run_practice(df, feature_cols, years, out_dir):
    """50-tree random forest over every window; writes predictions + window audit."""
    from sklearn.ensemble import RandomForestRegressor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = window_table(df, years)
    print_window_table(table)

    preds, fit_seconds = [], []
    for year in years:
        train_mask, test_mask = get_window(df, year)
        model = RandomForestRegressor(
            n_estimators=50, max_depth=3, max_features=0.05,
            random_state=42 + year, n_jobs=slurm_cpus(),
        )
        t0 = time.time()
        model.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, TARGET])
        fit_seconds.append(round(time.time() - t0, 2))

        y_pred = model.predict(df.loc[test_mask, feature_cols])

        chunk = df.loc[test_mask, [ID_COL, DATE_COL, TARGET]].copy()
        chunk["prediction_rf50"] = y_pred
        chunk["forecast_year"] = year
        preds.append(chunk)
        print(f"[{year}] fit {fit_seconds[-1]:.2f}s | train {train_mask.sum():>7,} | "
              f"test {test_mask.sum():>6,}")

    table["fit_seconds"] = fit_seconds
    predictions = pd.concat(preds, ignore_index=True)

    suffix = f"_{years[0]}" if len(years) == 1 else ""
    table.to_csv(out_dir / f"rolling_window_audit{suffix}.csv", index=False)
    predictions.to_csv(out_dir / f"task4_practice_predictions{suffix}.csv", index=False)
    print(f"\nWrote {len(predictions):,} predictions to {out_dir}/")


# ---------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 4 rolling-window forecasting loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_win = sub.add_parser("windows", help="print train/test dates and run the look-ahead check")
    p_win.add_argument("--data", required=True)
    p_win.add_argument("--years", nargs="+", type=int, default=FORECAST_YEARS)

    p_jobs = sub.add_parser("make-jobs", help="write the track x investor x year job list")
    p_jobs.add_argument("--tracks", nargs="+", required=True)
    p_jobs.add_argument("--investors", nargs="+", required=True)
    p_jobs.add_argument("--years", nargs="+", type=int, default=FORECAST_YEARS)
    p_jobs.add_argument("--out", default="config/jobs.csv")

    p_rf = sub.add_parser("practice", help="small-sample 50-tree random forest run")
    p_rf.add_argument("--data", required=True)
    p_rf.add_argument("--features-file", help="one feature name per line (default: all non-meta columns)")
    p_rf.add_argument("--years", nargs="+", type=int, default=FORECAST_YEARS)
    p_rf.add_argument("--out-dir", default="results/task4_practice")

    args = parser.parse_args(argv)

    if args.cmd == "windows":
        df = load_panel(args.data, columns=[ID_COL, DATE_COL, TARGET])
        df = df.dropna(subset=[TARGET])
        print_window_table(window_table(df, args.years))

    elif args.cmd == "make-jobs":
        jobs = build_job_list(args.tracks, args.investors, args.years)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        jobs.to_csv(args.out, index=False)
        print(f"Wrote {len(jobs)} jobs to {args.out}  ->  #SBATCH --array=0-{len(jobs) - 1}")

    elif args.cmd == "practice":
        df = load_panel(args.data)
        feature_cols = get_feature_cols(df, args.features_file)
        print(f"Loaded {len(df):,} rows, {df[DATE_COL].nunique()} months, "
              f"{len(feature_cols)} features")
        df = prepare_panel(df, feature_cols)
        run_practice(df, feature_cols, args.years, args.out_dir)


if __name__ == "__main__":
    main()
