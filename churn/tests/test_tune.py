from pathlib import Path

import mlflow

from churn.config import Settings
from churn.train import DEFAULT_PARAMS
from churn.tune import tune


def test_tune_returns_params_from_the_search_space(settings: Settings, raw_csv: Path) -> None:
    result = tune(settings, n_trials=3, data_path=raw_csv, cv_folds=3)

    assert result.n_trials == 3
    assert 0.5 < result.best_cv_roc_auc <= 1
    assert 100 <= result.best_params["n_estimators"] <= 800
    assert 0.005 <= result.best_params["learning_rate"] <= 0.2
    # Tuned params are a drop-in replacement for the defaults.
    assert set(result.best_params) == set(DEFAULT_PARAMS)


def test_every_trial_is_a_nested_mlflow_run(settings: Settings, raw_csv: Path) -> None:
    result = tune(settings, n_trials=3, data_path=raw_csv, cv_folds=3)

    runs = mlflow.search_runs(experiment_names=[settings.experiment_name], output_format="list")
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == result.run_id]
    assert len(children) == 3
    assert all("cv_roc_auc" in r.data.metrics for r in children)
    parent = next(r for r in runs if r.info.run_id == result.run_id)
    assert parent.data.metrics["best_cv_roc_auc"] == result.best_cv_roc_auc
    assert parent.data.tags["run_type"] == "tuning"
