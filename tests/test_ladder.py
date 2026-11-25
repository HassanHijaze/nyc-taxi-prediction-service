"""The feature ladder: exact feature sets, logged like the RF baseline, never a selection candidate.

Small made-up trips, MLflow replaced by a recorder (no server, no download).
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.train as train
from src import evaluation as ev
from src import selection
from src.baselines import LADDER_RUNS, LADDER_STAGE, RF_BASELINE_RUN
from src.features import FEATURES, build_features
from tests.fakes import fake_zones

ZONES_ONLY = ["PULocationID", "DOLocationID"]
ZONES_LOCATION = [*ZONES_ONLY, "PU_lat", "PU_lon", "DO_lat", "DO_lon", "straight_dist"]


def test_ladder_steps_use_exactly_the_listed_feature_sets():
    assert LADDER_RUNS == {"ladder-zones": ZONES_ONLY, "ladder-zones-location": ZONES_LOCATION}
    assert set(ZONES_LOCATION) < set(FEATURES)  # every step is a subset; rf-baseline (all) is the last


def january_trips(n_per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    zones = fake_zones().index.to_numpy()
    days = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    pickup = days.repeat(n_per_day) + pd.to_timedelta(np.tile(np.arange(n_per_day) * 3, len(days)), unit="h")
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "PULocationID": rng.choice(zones, len(pickup)),
            "DOLocationID": rng.choice(zones, len(pickup)),
            "duration": rng.uniform(2, 40, len(pickup)),
        }
    )
    return df


class MlflowRecorder:
    """Records each run's name, params, metrics and tags."""

    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.current: dict | None = None

    @contextmanager
    def start_run(self, run_name=None, **_):
        self.current = self.runs.setdefault(run_name, {"params": {}, "metrics": {}, "tags": {}})
        yield SimpleNamespace(info=SimpleNamespace(run_id=f"id-{run_name}"))
        self.current = None

    def log_params(self, params):
        self.current["params"].update(params)

    def log_metrics(self, metrics):
        self.current["metrics"].update(metrics)

    def set_tags(self, tags):
        self.current["tags"].update(tags)


@pytest.fixture
def logged(monkeypatch):
    """Run the RF baseline and the ladder; return (MLflow record, columns each RF was fitted on)."""
    recorder = MlflowRecorder()
    for name in ("start_run", "log_params", "log_metrics", "set_tags"):
        monkeypatch.setattr(train.mlflow, name, getattr(recorder, name))
    fitted: list[tuple[list[str], dict, int]] = []
    real_fit = ev.fit_rf

    def recording_fit(params, X, y):
        fitted.append((list(X.columns), dict(params), len(X)))
        return real_fit({**params, "n_estimators": 2}, X, y)  # tiny forest: fast test

    monkeypatch.setattr(ev, "fit_rf", recording_fit)

    splits = train.make_splits(january_trips())
    X_sample, X_val = build_features(splits.sample, fake_zones()), build_features(splits.val, fake_zones())
    train.log_rf_baseline(splits, X_sample, X_val)
    train.log_feature_ladder(splits, X_sample, X_val)
    return recorder.runs, fitted, len(splits.sample)


def test_ladder_runs_fit_and_log_exactly_their_feature_sets(logged):
    runs, fitted, sample_rows = logged
    assert list(runs) == [RF_BASELINE_RUN, "ladder-zones", "ladder-zones-location"]
    assert [cols for cols, _, _ in fitted] == [FEATURES, ZONES_ONLY, ZONES_LOCATION]
    # same RF settings and the same training sample for every step
    assert all(params == train.BASELINE_PARAMS and rows == sample_rows for _, params, rows in fitted)

    for name, features in LADDER_RUNS.items():
        run = runs[name]
        assert run["params"]["features"] == ",".join(features)
        assert {"val_rmse", "val_mae", "model_mb"} <= set(run["metrics"])
        assert run["tags"] == {"stage": LADDER_STAGE}
    assert runs[RF_BASELINE_RUN]["params"]["features"] == ",".join(FEATURES)
    assert "stage" not in runs[RF_BASELINE_RUN]["tags"]  # the RF baseline stays a selection candidate


def test_ladder_runs_are_not_selection_candidates(monkeypatch):
    import src.tracking as tracking

    def run(run_id, val_rmse, model_mb, tags=None):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id),
            data=SimpleNamespace(
                metrics={"val_rmse": val_rmse, "val_mae": val_rmse / 2, "model_mb": model_mb},
                params={"n_estimators": "50", "max_depth": "20", "min_samples_leaf": "5"},
                tags=tags or {},
            ),
        )

    # the ladder runs are better and smaller than everything else: they would win if they were read
    runs = {
        RF_BASELINE_RUN: run("rf1", 4.5, 80.0),
        "optuna-tuning": run("tune1", np.nan, np.nan),
        **{name: run(f"id-{name}", 1.0, 1.0, {"stage": LADDER_STAGE}) for name in LADDER_RUNS},
    }
    trial = {"run_id": "t0", "tags.mlflow.runName": "trial-0", "metrics.val_rmse": 4.4}
    trials = pd.DataFrame([{**trial, "metrics.val_mae": 2.8, "metrics.model_mb": 300.0}])
    requested: list[str] = []

    def latest_run(name, experiment=None):
        requested.append(name)
        return runs[name]

    monkeypatch.setattr(tracking, "latest_run", latest_run)
    monkeypatch.setattr(tracking, "child_runs", lambda run_id, experiment=None: trials)

    table = selection.collect_candidates()
    assert table["candidate"].tolist() == [RF_BASELINE_RUN, "trial-0"]
    assert not set(requested) & set(LADDER_RUNS)
    assert selection.apply_rule(table).iloc[0]["candidate"] == "trial-0"
