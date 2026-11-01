"""Train one investor for one year and save that year's forecasts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = ("accounting", "market")
FEATURE_KEYS = {
    "accounting": "accounting_table6_pure_fundamentals",
    "market": "market_table8_pure_market",
}


def parameters_for(mode: str) -> tuple[int, int, range]:
    """Training months, default number of trees, and valid forecast years."""
    if mode == "smoke":
        return 60, 50, range(2005, 2016)
    if mode == "60":
        return 60, 2000, range(2005, 2016)
    if mode == "120":
        return 120, 2000, range(2010, 2016)
    raise ValueError(f"Unknown experiment mode: {mode}")


def load_manifest(manifest_path: Path, track: str) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    features = manifest[FEATURE_KEYS[track]]
    if not features or len(set(features)) != len(features):
        raise ValueError("Expected nonempty unique feature list")
    if len(features) != manifest.get(f"{track}_count", len(features)):
        raise ValueError("Feature count mismatch")
    return features


def read_job_window(data_dir: Path, year: int, window: int, features: list[str],
                    stocks_per_month: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load the *complete required calendar window* from monthly Parquet files.

    For smoke tests only, sample AFTER preprocessing/ranking in the provided files.
    No test labels are used in feature selection, ranking or model fitting.
    """
    from src.rolling_window import load_panel, window_months

    first_train, _, first_test, last_test = window_months(year, window)
    # Keep unranked signal-month market value and the raw next-month return for
    # Task 7's value-weighted portfolios and backtest. Neither is a predictor.
    columns = ["id", "eom", "ret_exc_lead1m", "ret_lead1m", "me", *features]
    frames = []
    expected_periods = pd.period_range(first_train, last_test, freq="M")
    if stocks_per_month is not None:
        expected_periods = expected_periods[:window + 1]  # first forecast month only
    for j, period in enumerate(expected_periods):
        path = Path(data_dir) / f"{period}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Required data month missing: {path}")
        frame = load_panel(path, columns=columns)
        if frame.empty or not (frame.eom.dt.to_period("M") == period).all():
            raise ValueError(f"Empty/misdated data file: {path}")
        if frame[["id", "eom"]].duplicated().any():
            raise ValueError(f"Duplicate stock-month keys: {path}")
        if stocks_per_month is not None and len(frame) > stocks_per_month:
            frame = frame.sample(n=stocks_per_month, random_state=seed + j)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return df


def create_jobs(mode: str, output: Path, investors: int = 50,
                tracks: tuple[str, ...] = TRACKS) -> pd.DataFrame:
    from src.rolling_window import build_job_list

    months, _, years = parameters_for(mode)
    if investors < 1 or investors > 50:
        raise ValueError("investors must be between 1 and 50")
    if mode == "smoke":
        years = [2005]
    jobs = build_job_list(tracks, list(range(1, investors + 1)), years=years)
    jobs["mode"] = mode
    jobs["train_months"] = months
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(output, index=False)
    return jobs


def resolve_job(mode: str, job_file: Path | None, task_id: int | None,
                track: str | None, investor_id: int | None,
                forecast_year: int | None) -> tuple[str, int, int]:
    if job_file is not None:
        from src.rolling_window import job_for_task
        # job_for_task accepts SLURM_ARRAY_TASK_ID when task_id is None.
        entry = job_for_task(job_file, task_id)
        if "mode" in entry and str(entry["mode"]) != mode:
            raise ValueError(f"Jobs file mode {entry['mode']!r} != --mode {mode!r}")
        track, investor_id, forecast_year = (
            str(entry["track"]), int(entry["investor"]), int(entry["forecast_year"])
        )
    if track not in TRACKS:
        raise ValueError("Provide --track accounting|market or --jobs-csv")
    if investor_id is None or not 1 <= investor_id <= 50:
        raise ValueError("Provide --investor-id between 1 and 50")
    _, _, eligible = parameters_for(mode)
    if forecast_year not in eligible:
        raise ValueError(f"{mode}: forecast_year must be in {eligible.start}-{eligible.stop-1}")
    return track, investor_id, forecast_year


def run_job(*, data_dir: Path, manifest_path: Path, output_dir: Path,
            mode: str, track: str, investor_id: int, forecast_year: int,
            device: str = "cuda", n_estimators: int | None = None,
            n_jobs: int | None = None, seed_base: int = 1002,
            smoke_stocks_per_month: int = 80) -> tuple[Path, Path]:
    from src.investor_rf import train_one_investor_gpu
    from src.rolling_window import get_window, slurm_cpus

    train_months, default_trees, years = parameters_for(mode)
    if track not in TRACKS or investor_id not in range(1, 51) or forecast_year not in years:
        raise ValueError("Invalid track/investor/year combination")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if n_estimators is None:
        n_estimators = default_trees
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if n_jobs is None:
        n_jobs = slurm_cpus(default=1)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    features = load_manifest(manifest_path, track)
    panel = read_job_window(
        data_dir, forecast_year, train_months, features,
        stocks_per_month=smoke_stocks_per_month if mode == "smoke" else None,
        seed=seed_base,
    )
    train_mask, test_mask = get_window(panel, forecast_year, train_months=train_months)
    train = panel.loc[train_mask].reset_index(drop=True)
    test = panel.loc[test_mask].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Empty training/test window")
    # Stable investor feature subset across every forecast year: unlike a monthly seed,
    # seed never changes with forecast_year or mode.
    seed = seed_base + investor_id - 1
    output = train_one_investor_gpu(
        train, test, track, investor_id=investor_id, seed=seed,
        manifest_path=manifest_path, features_are_ranked=True,
        target_col="ret_exc_lead1m", n_estimators=n_estimators,
        max_depth=3, device=device, n_jobs=n_jobs,
    )
    forecast = output["predictions"].copy()
    forecast_rows = len(forecast)
    forecast = forecast.merge(
        test[["id", "eom", "ret_lead1m", "me"]],
        on=["id", "eom"], how="inner", validate="one_to_one",
    )
    if len(forecast) != forecast_rows:
        raise ValueError("Model forecasts do not match the prediction stock/month keys")
    forecast["forecast_year"] = forecast_year
    forecast["train_window"] = train_months
    forecast["mode"] = mode
    forecast["n_estimators"] = n_estimators
    if not np.isfinite(pd.to_numeric(forecast["prediction"], errors="coerce")).all():
        raise ValueError("Model returned missing or infinite predictions")

    folder = Path(output_dir) / "predictions" / mode / track / str(forecast_year)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"investor_{investor_id:03d}.csv"
    json_path = folder / f"investor_{investor_id:03d}.json"
    # Only one job should write the same path. SLURM job list has unique keys.
    forecast.to_csv(csv_path, index=False)
    metadata = dict(output["parameters"])
    metadata.update({"forecast_year": forecast_year, "train_window": train_months,
                     "mode": mode, "prediction_file": str(csv_path),
                     "expected_test_months": 1 if mode == "smoke" else 12})
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(forecast):,} predictions to {csv_path}")
    return csv_path, json_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SML one-investor-year experiment launcher")
    cmd = parser.add_subparsers(dest="command", required=True)

    jobs = cmd.add_parser("make-jobs", help="Make SLURM array CSV for both tracks")
    jobs.add_argument("--mode", choices=("smoke", "60", "120"), required=True)
    jobs.add_argument("--out", type=Path, required=True)
    jobs.add_argument("--investors", type=int)
    jobs.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))

    run = cmd.add_parser("run", help="Fit one track/investor/year and write predictions")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--manifest", type=Path, default=Path("config/feature_manifest.json"))
    run.add_argument("--output-dir", type=Path, default=Path("results/daehyun"))
    run.add_argument("--mode", choices=("smoke", "60", "120"), required=True)
    run.add_argument("--jobs-csv", type=Path, help="Job list generated by make-jobs")
    run.add_argument("--task-id", type=int, help="0-based job array task ID")
    run.add_argument("--track", choices=TRACKS)
    run.add_argument("--investor-id", type=int)
    run.add_argument("--forecast-year", type=int)
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--n-estimators", type=int)
    run.add_argument("--n-jobs", type=int)
    run.add_argument("--seed-base", type=int, default=1002)
    run.add_argument("--smoke-stocks-per-month", type=int, default=80)

    args = parser.parse_args(argv)
    if args.command == "make-jobs":
        n = args.investors or (1 if args.mode == "smoke" else 50)
        table = create_jobs(args.mode, args.out, n, tuple(args.tracks))
        print(f"Wrote {len(table)} array tasks to {args.out}; --array=0-{len(table)-1}")
        return

    track, investor_id, forecast_year = resolve_job(
        args.mode, args.jobs_csv, args.task_id,
        args.track, args.investor_id, args.forecast_year,
    )
    if args.smoke_stocks_per_month <= 0:
        parser.error("--smoke-stocks-per-month must be positive")
    run_job(
        data_dir=args.data_dir, manifest_path=args.manifest,
        output_dir=args.output_dir, mode=args.mode, track=track,
        investor_id=investor_id, forecast_year=forecast_year,
        device=args.device or ("cpu" if args.mode == "smoke" else "cuda"),
        n_estimators=args.n_estimators, n_jobs=args.n_jobs,
        seed_base=args.seed_base,
        smoke_stocks_per_month=args.smoke_stocks_per_month,
    )


if __name__ == "__main__":
    main()
