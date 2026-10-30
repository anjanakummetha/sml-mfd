"""Aggregate investor forecasts, evaluate OOS returns, and compute MFD gap.

MFD = |mean(accounting investor predictions) - mean(market investor predictions)|.
The within-track investor SD is reported separately, NOT called MFD.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["id", "eom"]
TRACKS = ("accounting", "market")


def aggregate_one_track(files: list[Path], expected_investors: int) -> pd.DataFrame:
    """Require identical (stock, month) grids and realized labels for all investors."""
    if len(files) != expected_investors:
        raise ValueError(f"Expected {expected_investors} files, got {len(files)}: {files[:2]}")
    frame0 = None
    predictions = []
    seen_ids = set()
    for path in sorted(files):
        frame = pd.read_csv(path, parse_dates=["eom"]).sort_values(KEYS).reset_index(drop=True)
        required = set(KEYS + ["track", "investor_id", "ret_exc_lead1m", "prediction",
                               "mode", "train_window", "forecast_year"])
        if required - set(frame):
            raise ValueError(f"Missing required columns in {path}: {required - set(frame)}")
        if frame[KEYS].duplicated().any():
            raise ValueError(f"Duplicate stock/month keys in {path}")
        inv = frame["investor_id"].unique()
        if len(inv) != 1 or inv[0] in seen_ids:
            raise ValueError(f"Missing/duplicate investor ID: {path}")
        seen_ids.add(inv[0])
        if not np.isfinite(pd.to_numeric(frame.prediction, errors="coerce")).all():
            raise ValueError(f"Nonfinite prediction in {path}")
        if frame0 is None:
            frame0 = frame
            for name in ("track", "mode", "train_window", "forecast_year"):
                if frame0[name].nunique() != 1:
                    raise ValueError(f"Mixed {name} in {path}")
        else:
            if not frame[KEYS].equals(frame0[KEYS]):
                raise ValueError(f"Stock/month alignment mismatch in {path}")
            for name in ("track", "mode", "train_window", "forecast_year"):
                if not (frame[name].to_numpy() == frame0[name].to_numpy()).all():
                    raise ValueError(f"{name} does not match between investors")
            first_y = pd.to_numeric(frame0.ret_exc_lead1m, errors="coerce").to_numpy(float)
            next_y = pd.to_numeric(frame.ret_exc_lead1m, errors="coerce").to_numpy(float)
            if not np.allclose(first_y, next_y, rtol=1e-6, atol=1e-10, equal_nan=True):
                raise ValueError(f"Realized return mismatch in {path}")
        predictions.append(frame.prediction.to_numpy(dtype=float))
    stacked = np.stack(predictions)
    result = frame0[["id", "eom", "ret_exc_lead1m", "track", "mode", "train_window", "forecast_year"]].copy()
    result = result.rename(columns={"ret_exc_lead1m": "actual_excess_return"})
    result["predicted_excess_return"] = stacked.mean(axis=0)
    result["investor_disagreement_sd"] = stacked.std(axis=0, ddof=0)
    result["n_investors"] = len(files)
    return result


def load_aggregates(output_dir: Path, mode: str, expected_investors: int) -> pd.DataFrame:
    """Process one (year, track) at a time to bound memory on the cluster."""
    directory = Path(output_dir) / "predictions" / mode
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows = []
    for track in TRACKS:
        for year_dir in sorted((directory / track).glob("[0-9][0-9][0-9][0-9]")):
            files = sorted(year_dir.glob("investor_*.csv"))
            if files:
                rows.append(aggregate_one_track(files, expected_investors))
    if not rows:
        raise ValueError(f"No predictions found for {mode} in {directory}")
    # Evaluation is an end-of-run operation. Do not silently score a partial
    # cluster run or mistake missing tracks/years for missing stock forecasts.
    expected_years = ([2005] if mode == "smoke"
                      else list(range(2005, 2016)) if mode == "60"
                      else list(range(2010, 2016)))
    actual_jobs = {(str(row.track.iloc[0]), int(row.forecast_year.iloc[0])) for row in rows}
    required_jobs = {(track, year) for track in TRACKS for year in expected_years}
    if actual_jobs != required_jobs:
        missing = sorted(required_jobs - actual_jobs)
        extra = sorted(actual_jobs - required_jobs)
        raise ValueError(f"Incomplete or unexpected track/year jobs. Missing={missing[:8]}, Extra={extra[:8]}")
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(Path(output_dir) / f"aggregated_{mode}.csv", index=False)
    return result


def compute_mfd(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Inner-join track consensus forecasts by stock, signal month, and window."""
    if not set(TRACKS).issubset(set(aggregate.track)):
        return pd.DataFrame(columns=["id", "eom", "mode", "train_window", "mfd"])
    key = ["id", "eom", "mode", "train_window"]
    outputs = ["predicted_excess_return", "actual_excess_return", "investor_disagreement_sd"]
    a = aggregate.loc[aggregate.track == "accounting", key + outputs].copy()
    m = aggregate.loc[aggregate.track == "market", key + outputs].copy()
    if a[key].duplicated().any() or m[key].duplicated().any():
        raise ValueError("Duplicate consensus forecast keys")
    combined = a.merge(m, on=key, how="inner", validate="one_to_one", suffixes=("_accounting", "_market"))
    if combined.empty:
        raise ValueError("Accounting and market forecasts have no overlapping stock/month keys")
    if len(combined) != len(a) or len(combined) != len(m):
        raise ValueError("Accounting and market forecasts must cover identical stock/month keys")
    aa = combined.actual_excess_return_accounting.to_numpy(float)
    mm = combined.actual_excess_return_market.to_numpy(float)
    if not np.allclose(aa, mm, rtol=1e-6, atol=1e-10, equal_nan=True):
        raise ValueError("Actual returns differ across tracks")
    combined["signed_gap"] = (combined.predicted_excess_return_accounting
                              - combined.predicted_excess_return_market)
    combined["mfd"] = combined.signed_gap.abs()
    combined = combined.rename(columns={
        "predicted_excess_return_accounting": "accounting_mean",
        "predicted_excess_return_market": "market_mean",
        "investor_disagreement_sd_accounting": "accounting_disagreement_sd",
        "investor_disagreement_sd_market": "market_disagreement_sd",
        "actual_excess_return_accounting": "actual_excess_return",
    }).drop(columns=["actual_excess_return_market"])
    return combined


def _accuracy(frame: pd.DataFrame) -> dict:
    actual = frame.actual_excess_return.to_numpy(dtype=float)
    predicted = frame.predicted_excess_return.to_numpy(dtype=float)
    error = actual - predicted
    sse = float((error ** 2).sum())
    sst0 = float((actual ** 2).sum())
    return {
        "n_predictions": len(frame),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mae": float(np.abs(error).mean()),
        "zero_benchmark_rmse": float(np.sqrt((actual ** 2).mean())),
        "oos_r_squared": float(1 - sse / sst0) if sst0 > 0 else float("nan"),
    }


def evaluate_predictions(aggregates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall and monthly OOS accuracy per track/mode/window; missing labels excluded."""
    required = {"eom", "track", "mode", "train_window", "actual_excess_return", "predicted_excess_return"}
    if required - set(aggregates):
        raise ValueError(f"Missing columns: {sorted(required - set(aggregates))}")
    valid = aggregates.copy()
    for col in ("actual_excess_return", "predicted_excess_return"):
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.loc[np.isfinite(valid.actual_excess_return)
                      & np.isfinite(valid.predicted_excess_return)]
    if valid.empty:
        raise ValueError("No finite out-of-sample labels and predictions")
    overall_rows, monthly_rows = [], []
    for keys, subset in valid.groupby(["mode", "train_window", "track"], sort=True):
        labels = dict(zip(["mode", "train_window", "track"], keys))
        monthly_ics = []
        for month, frame in subset.groupby("eom", sort=True):
            ic = (frame.actual_excess_return.corr(frame.predicted_excess_return, method="spearman")
                  if len(frame) >= 3 and frame.actual_excess_return.nunique() > 1
                  and frame.predicted_excess_return.nunique() > 1 else float("nan"))
            row = {**labels, "eom": month, **_accuracy(frame), "spearman_ic": ic}
            monthly_rows.append(row)
            if np.isfinite(ic):
                monthly_ics.append(ic)
        ics = np.asarray(monthly_ics, dtype=float)
        ic_sd = float(ics.std(ddof=1)) if len(ics) > 1 else float("nan")
        overall_rows.append({
            **labels, **_accuracy(subset), "n_months": subset.eom.nunique(),
            "mean_monthly_spearman_ic": float(ics.mean()) if len(ics) else float("nan"),
            "positive_ic_month_fraction": float((ics > 0).mean()) if len(ics) else float("nan"),
            "simple_ic_t_stat": float(ics.mean() / ic_sd * np.sqrt(len(ics)))
                if ic_sd > 0 else float("nan"),
        })
    return pd.DataFrame(overall_rows), pd.DataFrame(monthly_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate investors and evaluate OOS accuracy")
    parser.add_argument("--output-dir", type=Path, default=Path("results/daehyun"))
    parser.add_argument("--mode", choices=("smoke", "60", "120"), required=True)
    parser.add_argument("--expected-investors", type=int)
    args = parser.parse_args(argv)
    count = args.expected_investors or (1 if args.mode == "smoke" else 50)
    if count < 1:
        parser.error("--expected-investors must be positive")
    aggregated = load_aggregates(args.output_dir, args.mode, count)
    overall, monthly = evaluate_predictions(aggregated)
    overall.to_csv(args.output_dir / f"accuracy_overall_{args.mode}.csv", index=False)
    monthly.to_csv(args.output_dir / f"accuracy_monthly_{args.mode}.csv", index=False)
    mfd = compute_mfd(aggregated)
    if not mfd.empty:
        mfd.to_csv(args.output_dir / f"mfd_{args.mode}.csv", index=False)
    print(overall.to_string(index=False))
    print(f"MFD rows: {len(mfd)}")


if __name__ == "__main__":
    main()
