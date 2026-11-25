"""Data, splits and the `nyc-taxi train` step (baselines).

Splits (fixed, January 2025 only; nothing is ever retrained on more data):
    train       Jan 1-20    every model learns from these days only
    validation  Jan 21-25   comparing models (baselines, RF baseline, Optuna trials)
    test        Jan 26-31   used once, by `nyc-taxi select`, for the winner and the baselines

Random Forests (RF baseline, Optuna trials, the selected winner) are fitted on the same
training sample: SAMPLE_SIZE trips drawn from the training days with a fixed seed
(training_sample), so a run can be refitted exactly. The naive baselines use all
training-day trips for their statistics.

`nyc-taxi train` logs one MLflow run each, scored on all validation trips:
    baseline-overall-median     median duration of the training trips   (src/baselines.py)
    baseline-zone-pair-median   median per zone pair, unseen pairs get the overall median
    rf-baseline                 Random Forest with BASELINE_PARAMS (also logs model_mb)
    ladder-zones                feature ladder (tag stage=ladder): BASELINE_PARAMS on the zone IDs only
    ladder-zones-location       zone IDs + zone centers + straight-line distance; the last ladder step
                                is rf-baseline itself (all features). Ladder runs are not selection
                                candidates (src/baselines.py LADDER_RUNS)

Nothing is registered here: `nyc-taxi select` registers the winner.

Run from the project root:  uv run nyc-taxi train   (or uv run python -m src.train)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from src import evaluation as ev
from src.baselines import LADDER_RUNS, LADDER_STAGE, NAIVE_BASELINES, RF_BASELINE_RUN
from src.experiment import EXPERIMENT, TRACKING_URI
from src.features import FEATURES, build_features, build_zone_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_PATH = ROOT / "data" / "yellow_tripdata_2025-01.parquet"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
BASELINE_RUN_NAME = RF_BASELINE_RUN  # "rf-baseline"
SAMPLE_SIZE = 500_000  # training-day trips used to fit every Random Forest

BASELINE_PARAMS = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}


def download_january() -> None:
    """Download the January parquet file once, keeping only the columns we use.

    The file is written to a temporary name first and renamed when complete, so an
    interrupted download never leaves a broken file under DATA_PATH.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".parquet.tmp")
    pd.read_parquet(DATA_URL, columns=COLUMNS).to_parquet(tmp_path)
    tmp_path.replace(DATA_PATH)  # atomic rename


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Keep January 2025 pickups, sort by pickup time, add duration (minutes), keep 1-60 minute trips."""
    pickup = df["tpep_pickup_datetime"]
    df = df[(pickup >= "2025-01-01") & (pickup < "2025-02-01")]
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df[(df["duration"] >= 1) & (df["duration"] <= 60)]


def load_january() -> pd.DataFrame:
    """All January 2025 trips of 1-60 minutes, sorted by pickup time."""
    if not DATA_PATH.exists():
        download_january()
    return clean_trips(pd.read_parquet(DATA_PATH, columns=COLUMNS))


def load_data(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Random sample of n January trips, back in time order (used by the model tests)."""
    return load_january().sample(n=n, random_state=42).sort_index()


def split_by_time(df: pd.DataFrame):
    """Boolean masks: train Jan 1-20, validation Jan 21-25, test Jan 26-31."""
    pickup = df["tpep_pickup_datetime"]
    train = pickup < "2025-01-21"
    val = (pickup >= "2025-01-21") & (pickup < "2025-01-26")
    test = pickup >= "2025-01-26"
    return train, val, test


def training_sample(train_df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """The trips every Random Forest is fitted on: n training-day trips (fixed seed), in time order."""
    if len(train_df) > n:
        train_df = train_df.sample(n=n, random_state=42)
    return train_df.sort_index()


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame  # all Jan 1-20 trips (statistics of the naive baselines)
    sample: pd.DataFrame  # training sample: what every Random Forest is fitted on
    val: pd.DataFrame  # all Jan 21-25 trips
    test: pd.DataFrame  # all Jan 26-31 trips


def make_splits(full: pd.DataFrame) -> Splits:
    tr, va, te = split_by_time(full)
    return Splits(train=full[tr], sample=training_sample(full[tr]), val=full[va], test=full[te])


def load_splits() -> Splits:
    return make_splits(load_january())


def period_params(splits: Splits) -> dict[str, str | int]:
    """Periods and sizes of the data a run used, as MLflow params."""
    params: dict[str, str | int] = {}
    for name in ("train", "sample", "val", "test"):
        df = getattr(splits, name)
        pickup = df["tpep_pickup_datetime"]
        params[f"{name}_period"] = f"{pickup.min():%Y-%m-%d}..{pickup.max():%Y-%m-%d}"
        params[f"{name}_rows"] = len(df)
    return params


def log_naive_baselines(splits: Splits) -> dict[str, dict[str, float]]:
    """One MLflow run per naive baseline: training-day statistics, scored on validation."""
    results = {}
    for run_name, (predict, description) in NAIVE_BASELINES.items():
        pred = predict(splits.train, splits.val)
        metrics = ev.score_predictions("val", splits.train, splits.val, pred)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"baseline": description, **period_params(splits)})
            mlflow.log_metrics(metrics)
        print(run_name, {k: round(metrics[k], 4) for k in ("val_rmse", "val_mae")})
        results[run_name] = metrics
    return results


def _log_rf_run(
    run_name: str,
    features: list[str],
    splits: Splits,
    X_sample: pd.DataFrame,
    X_val: pd.DataFrame,
    tags: dict[str, str] | None = None,
) -> dict[str, float]:
    """BASELINE_PARAMS fitted on the training sample (only `features`), scored on validation."""
    model = ev.fit_rf(BASELINE_PARAMS, X_sample[features], splits.sample["duration"])
    pred = model.predict(X_val[features])
    metrics = ev.score_predictions("val", splits.train, splits.val, pred)
    metrics["model_mb"] = ev.model_size_mb(model)
    del model
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**BASELINE_PARAMS, "features": ",".join(features), **period_params(splits)})
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
    print(run_name, {k: round(metrics[k], 4) for k in ("val_rmse", "val_mae", "model_mb")})
    return metrics


def log_rf_baseline(splits: Splits, X_sample: pd.DataFrame, X_val: pd.DataFrame) -> dict[str, float]:
    """rf-baseline: BASELINE_PARAMS on all FEATURES (a selection candidate)."""
    return _log_rf_run(BASELINE_RUN_NAME, FEATURES, splits, X_sample, X_val)


def log_feature_ladder(
    splits: Splits, X_sample: pd.DataFrame, X_val: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """One run per LADDER_RUNS step (tag stage=ladder); never a selection candidate."""
    return {
        run_name: _log_rf_run(run_name, features, splits, X_sample, X_val, tags={"stage": LADDER_STAGE})
        for run_name, features in LADDER_RUNS.items()
    }


def main() -> dict:
    """Log the three baseline runs and the feature-ladder runs (validation scores only)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)  # fails fast if the MLflow server is not running

    splits = load_splits()
    zone_xy = build_zone_xy(ROOT / "data")
    print("train:     ", ev.describe_period(splits.train), f"(sample: {len(splits.sample):,} trips)")
    print("validation:", ev.describe_period(splits.val))

    log_naive_baselines(splits)
    X_sample, X_val = build_features(splits.sample, zone_xy), build_features(splits.val, zone_xy)
    log_rf_baseline(splits, X_sample, X_val)
    log_feature_ladder(splits, X_sample, X_val)
    return {}


if __name__ == "__main__":
    main()
