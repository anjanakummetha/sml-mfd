"""One-investor XGBoost GPU random forest for the SML two-track project.

No file I/O or model training takes place on import. The caller owns rolling
windows, annual refitting, Parquet loading, SLURM scheduling and persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

TRACK_KEYS = {
    "accounting": "accounting_table6_pure_fundamentals",
    "market": "market_table8_pure_market",
}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "config" / "feature_manifest.json"


def load_track_features(track, manifest_path=DEFAULT_MANIFEST):
    """Load an exact, ordered track list; fail rather than silently drop columns."""
    if track not in TRACK_KEYS:
        raise ValueError("track must be 'accounting' or 'market'")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    features = manifest[TRACK_KEYS[track]]
    if (not isinstance(features, list) or not features
            or not all(isinstance(x, str) for x in features)
            or len(features) != len(set(features))):
        raise ValueError("Manifest must contain a nonempty list of unique feature names")
    if manifest.get(f"{track}_count", len(features)) != len(features):
        raise ValueError("Manifest count does not match the track list")
    return features


def select_investor_features(track_features, seed):
    """Select ceil(n/2) features using a seed and the manifest's original order.

    This preserves the Notebook's odd-count convention: 76 -> 38, 51 -> 26.
    Reuse the same seed/manifest across rolling windows to keep information fixed.
    """
    rng = np.random.default_rng(seed)
    count = max(1, (len(track_features) + 1) // 2)
    return sorted(rng.choice(track_features, count, replace=False).tolist())


def _months(frame, date_col):
    if frame.empty:
        raise ValueError("Input frame must not be empty")
    if date_col not in frame:
        raise ValueError(f"Missing date column: {date_col}")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid or missing signal dates")
    return dates.dt.to_period("M")


def _numeric(frame, columns):
    return (frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan))


def rank_transform_features(frame, feature_columns, date_col="eom"):
    """Transform RAW features within each full monthly cross-section.

    Manifest rule: 2*(average_rank-1)/(nonmissing_count-1)-1; missing,
    infinite or singleton values -> 0; float32. No target is used.
    Call before filtering rows by target availability or sampling stocks.
    Do not call on already transformed monthly Parquet inputs.
    """
    months = _months(frame, date_col)
    numeric = _numeric(frame, feature_columns)
    group = numeric.groupby(months, sort=False)
    ranks = group.rank(method="average")
    counts = group.transform("count")
    return (2 * (ranks - 1) / (counts - 1).replace(0, np.nan) - 1).fillna(0).astype("float32")


def train_one_investor_gpu(
    train_frame,
    prediction_frame,
    track,
    investor_id=1,
    seed=1002,
    *,
    manifest_path=DEFAULT_MANIFEST,
    features_are_ranked=True,
    target_col="ret_exc_lead1m",
    id_col="id",
    date_col="eom",
    n_estimators=2000,
    max_depth=3,
    max_samples=0.05,
    max_features=0.05,
    winsor_tail=0.01,
    device="cuda",
    reg_lambda=1e-5,
    n_jobs=-1,
):
    """Train one investor and return model, predictions, feature list and metadata.

    Inputs are pandas DataFrames. Defaults expect pre-ranked team Parquet data;
    use features_are_ranked=False for the raw practice CSV. Predict labels are
    optional and are never used to fit or preprocess the model.

    All training signal months must precede the FIRST prediction signal month.
    A label for signal month s is realized in s+1; formation is assumed to occur
    after the first prediction month's close, so those labels are observable.
    No target shift is applied. The caller must respect actual data availability.

    GPU fallback is rejected. device='cpu' is an explicit development option.
    XGBoost >= 2 is required. Row sampling is without replacement, unlike
    sklearn bootstrap.
    """
    import xgboost as xgb

    if int(xgb.__version__.split(".")[0]) < 2:
        raise RuntimeError("XGBoost >= 2.0 is required for the device parameter")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'")
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if not (0 < max_samples <= 1 and 0 < max_features <= 1):
        raise ValueError("Sampling fractions must be in (0, 1]")
    if not 0 <= winsor_tail < 0.5:
        raise ValueError("winsor_tail must be in [0, 0.5)")
    if not train_frame.index.is_unique or not prediction_frame.index.is_unique:
        raise ValueError("Reset frame indices before passing concatenated monthly data")

    features = load_track_features(track, manifest_path)
    for label, frame in [("train", train_frame), ("prediction", prediction_frame)]:
        missing = set(features + [id_col, date_col]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")
        if frame[id_col].isna().any() or frame.duplicated([id_col, date_col]).any():
            raise ValueError(f"Missing IDs or duplicate stock/date keys in {label}")
    if target_col not in train_frame:
        raise ValueError(f"Missing training target: {target_col}")
    if target_col in features or id_col in features or date_col in features:
        raise ValueError("Manifest includes a target, identifier or date as a feature")
    train_months = _months(train_frame, date_col)
    predict_months = _months(prediction_frame, date_col)
    if train_months.max() >= predict_months.min():
        raise ValueError("All training signal months must precede prediction months")

    chosen = select_investor_features(features, seed)
    preprocessing_start = perf_counter()
    if features_are_ranked:
        X_train = _numeric(train_frame, chosen).fillna(0).astype("float32")
        X_predict = _numeric(prediction_frame, chosen).fillna(0).astype("float32")
        if (X_train.abs().to_numpy().max() > 1.00001
                or X_predict.abs().to_numpy().max() > 1.00001):
            raise ValueError("Expected ranked values in [-1,1]; use features_are_ranked=False for raw data")
    else:
        X_train = rank_transform_features(train_frame, chosen, date_col)
        X_predict = rank_transform_features(prediction_frame, chosen, date_col)
    y = pd.to_numeric(train_frame[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    if not valid.any():
        raise ValueError("No finite training labels")
    X_train = X_train.loc[valid]
    y = y.loc[valid].astype("float32")
    lower, upper = y.quantile([winsor_tail, 1 - winsor_tail]).tolist()
    y = y.clip(lower, upper)
    preprocessing_seconds = perf_counter() - preprocessing_start

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1,
        num_parallel_tree=n_estimators, learning_rate=1.0,
        max_depth=max_depth, subsample=max_samples,
        sampling_method="uniform", colsample_bynode=max_features,
        colsample_bytree=1.0, colsample_bylevel=1.0,
        reg_lambda=reg_lambda, tree_method="hist", device=device,
        random_state=seed, n_jobs=n_jobs,
    )
    total_start = perf_counter()
    fit_start = perf_counter()
    model.fit(X_train, y)
    fit_seconds = perf_counter() - fit_start
    actual_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if device == "cuda" and not actual_device.startswith("cuda"):
        raise RuntimeError(f"Requested GPU but XGBoost used {actual_device}; check GPU allocation and environment")
    if model.get_booster().num_boosted_rounds() != 1:
        raise RuntimeError("Expected exactly one boosting round")
    # DMatrix prediction handles host input explicitly, avoiding the inplace
    # predictor's device-mismatch fallback. Includes input preparation/transfer.
    predict_start = perf_counter()
    predicted = model.get_booster().predict(xgb.DMatrix(X_predict))
    predict_seconds = perf_counter() - predict_start
    elapsed = perf_counter() - total_start
    if len(predicted) != len(prediction_frame) or not np.isfinite(predicted).all():
        raise RuntimeError("Incomplete or nonfinite predictions")

    columns = [id_col, date_col] + ([target_col] if target_col in prediction_frame else [])
    output = prediction_frame.loc[:, columns].copy()
    output["track"] = track
    output["investor_id"] = investor_id
    output["seed"] = seed
    output["prediction"] = predicted
    metadata = {
        "backend": "xgboost_rf", "xgboost_version": xgb.__version__,
        "requested_device": device, "actual_device": actual_device,
        "track": track, "investor_id": investor_id, "seed": seed,
        "n_estimators": n_estimators, "boosting_rounds": 1,
        "max_depth": max_depth, "max_samples": max_samples,
        "max_features": max_features, "reg_lambda": reg_lambda,
        "features_are_ranked": features_are_ranked,
        "n_candidate_features": len(features), "n_selected_features": len(chosen),
        "n_train_rows": len(y), "n_prediction_rows": len(output),
        "train_signal_start": str(train_months.min()),
        "train_signal_end": str(train_months.max()),
        "prediction_signal_start": str(predict_months.min()),
        "prediction_signal_end": str(predict_months.max()),
        "winsor_tail": winsor_tail, "winsor_bounds": [float(lower), float(upper)],
        "preprocessing_seconds": preprocessing_seconds,
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "fit_and_predict_seconds": elapsed,
        "row_sampling": "uniform_without_replacement",
    }
    return {
        "model": model, "predictions": output, "selected_features": chosen,
        "fit_and_predict_seconds": elapsed, "parameters": metadata,
    }
