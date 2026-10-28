"""Unit tests of Daehyun's integration contract and accuracy calculations."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment_runner import parameters_for, resolve_job, run_job
from src.evaluate_oos import aggregate_one_track, compute_mfd, evaluate_predictions


def test_modes_and_years():
    assert (parameters_for("60")[0], parameters_for("120")[0]) == (60, 120)
    assert len(parameters_for("60")[2]) == 11
    assert len(parameters_for("120")[2]) == 6
    assert resolve_job("60", None, None, "accounting", 1, 2005) == ("accounting", 1, 2005)
    with pytest.raises(ValueError):
        resolve_job("120", None, None, "market", 1, 2005)


def test_aggregation_26_market_features_not_25(tmp_path):
    base = pd.DataFrame({"id": [1, 2], "eom": ["2005-01-31", "2005-01-31"],
        "ret_exc_lead1m": [0.05, -0.05], "track": "market", "mode": "smoke",
        "train_window": 60, "forecast_year": 2005})
    files = []
    for i, predictions in [(1, [0.1, -0.1]), (2, [0.0, -0.2])]:
        file = tmp_path / f"investor_{i:03d}.csv"
        frame = base.assign(investor_id=i, prediction=predictions)
        frame.to_csv(file, index=False)
        files.append(file)
    aggregated = aggregate_one_track(files, expected_investors=2)
    assert aggregated.predicted_excess_return.tolist() == pytest.approx([0.05, -0.15])
    assert aggregated.investor_disagreement_sd.tolist() == pytest.approx([0.05, 0.05])
    with pytest.raises(ValueError, match="Expected 3 files"):
        aggregate_one_track(files, expected_investors=3)


def test_mfd_is_cross_track_gap_not_within_track_sd():
    base = pd.DataFrame({"id": [1, 2, 3], "eom": ["2005-01"] * 3,
        "mode": ["60"] * 3, "train_window": [60] * 3,
        "actual_excess_return": [.1, .2, -.1]})
    a = base.assign(track="accounting", predicted_excess_return=[0.1, 0.1, 0.1])
    m = base.assign(track="market", predicted_excess_return=[0.2, 0.0, 0.3])
    mfd = compute_mfd(pd.concat([a, m], ignore_index=True))
    assert mfd.mfd.tolist() == pytest.approx([0.1, 0.1, 0.2])
    overall, monthly = evaluate_predictions(pd.concat([a, m], ignore_index=True))
    assert len(overall) == 2 and len(monthly) == 2


def test_missing_future_labels_excluded_from_oos_metrics():
    data = pd.DataFrame({"id": [1, 2, 3], "eom": ["2015-12"] * 3,
        "track": "accounting", "mode": "60", "train_window": 60,
        "actual_excess_return": [0.1, np.nan, -0.1],
        "predicted_excess_return": [0.0, 0.0, 0.0]})
    overall, _ = evaluate_predictions(data)
    assert overall.n_predictions.iloc[0] == 2
    assert overall.oos_r_squared.iloc[0] == pytest.approx(0.0)


def test_run_job_reuses_both_team_modules(monkeypatch, tmp_path):
    """Contract test: mock functions with exact names/signatures from both PRs."""
    from src import experiment_runner as runner
    track = "accounting"
    features = ["f0", "f1", "f2", "f3"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"accounting_count": 4, "accounting_table6_pure_fundamentals": ["f0", "f1", "f2", "f3"]}')
    periods = pd.period_range("2000-01", periods=61, freq="M")
    panel = pd.DataFrame({"id": [1] * len(periods),
        "eom": periods.to_timestamp("M"),
        "ret_exc_lead1m": np.linspace(.01, .03, len(periods)),
        **{f: np.zeros(len(periods)) for f in features}})
    monkeypatch.setattr(runner, "read_job_window", lambda *args, **kwargs: panel)
    def get_window(df, forecast_year, train_months=60):
        months = df.eom.dt.to_period("M")
        return months < pd.Period("2005-01"), months == pd.Period("2005-01")
    fake_window = types.ModuleType("src.rolling_window")
    fake_window.get_window = get_window
    fake_window.slurm_cpus = lambda default=1: 1
    fake_investor = types.ModuleType("src.investor_rf")
    seen = []
    def train_one_investor_gpu(train, test, track, investor_id, seed, **kwargs):
        seen.append((len(train), len(test), track, investor_id, seed, kwargs))
        out = test[["id", "eom", "ret_exc_lead1m"]].copy()
        out["track"] = track
        out["investor_id"] = investor_id
        out["seed"] = seed
        out["prediction"] = 0.05
        return {"predictions": out, "parameters": {"backend": "xgboost_rf", "n_estimators": kwargs["n_estimators"]}}
    fake_investor.train_one_investor_gpu = train_one_investor_gpu
    monkeypatch.setitem(sys.modules, "src.rolling_window", fake_window)
    monkeypatch.setitem(sys.modules, "src.investor_rf", fake_investor)
    out, metadata = run_job(data_dir=tmp_path, manifest_path=manifest_path,
        output_dir=tmp_path, mode="smoke", track=track, investor_id=1, forecast_year=2005)
    assert seen[0][0:5] == (60, 1, track, 1, 1002)
    assert seen[0][5]["n_estimators"] == 50
    assert seen[0][5]["features_are_ranked"] is True
    assert len(pd.read_csv(out)) == 1 and metadata.exists()


def test_slurm_job_lists_use_matthew_contract(monkeypatch, tmp_path):
    from src.experiment_runner import create_jobs
    import itertools

    def build_job_list(tracks, investors, years):
        return pd.DataFrame([
            {"task_id": i, "track": t, "investor": inv, "forecast_year": y}
            for i, (t, inv, y) in enumerate(itertools.product(tracks, investors, years))
        ])

    fake_window = types.ModuleType("src.rolling_window")
    fake_window.build_job_list = build_job_list
    monkeypatch.setitem(sys.modules, "src.rolling_window", fake_window)
    full = create_jobs("60", tmp_path / "jobs_60.csv")
    backup = create_jobs("120", tmp_path / "jobs_120.csv")
    assert len(full) == 1100 and len(backup) == 600
    assert full.task_id.tolist() == list(range(1100))
    assert set(full.track) == {"accounting", "market"}
    assert full.forecast_year.min() == 2005 and backup.forecast_year.min() == 2010
    assert len(pd.read_csv(tmp_path / "jobs_60.csv")) == 1100
