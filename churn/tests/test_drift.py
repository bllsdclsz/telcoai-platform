import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from churn.drift import detect_drift, load_predictions
from churn.schema import FEATURE_COLUMNS
from churn.simulate import shift

from .conftest import make_customers


def test_same_distribution_does_not_drift() -> None:
    reference, current = make_customers(n=2000, seed=0), make_customers(n=500, seed=1)
    result = detect_drift(reference, current)
    assert not result.dataset_drift
    assert result.drifted_columns == []
    assert set(result.column_scores) == set(FEATURE_COLUMNS)


def test_shift_scenario_drifts_and_writes_report(tmp_path: Path) -> None:
    reference = make_customers(n=2000, seed=0)
    current = shift(make_customers(n=500, seed=1))
    report = tmp_path / "drift.html"

    result = detect_drift(reference, current, html_report=report)

    assert result.dataset_drift
    assert {"tenure", "MonthlyCharges", "Contract"} <= set(result.drifted_columns)
    assert report.stat().st_size > 0


def write_log(path: Path, customers: pd.DataFrame, scored_at: datetime) -> None:
    rows = customers[FEATURE_COLUMNS].to_dict(orient="records")
    lines = [json.dumps({**r, "scored_at": scored_at.isoformat()}, default=int) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_load_predictions_keeps_only_the_window(tmp_path: Path) -> None:
    now = datetime(2026, 9, 26, tzinfo=UTC)
    write_log(tmp_path / "predictions-2026-09-01.jsonl", make_customers(n=5), now - timedelta(25))
    write_log(tmp_path / "predictions-2026-09-25.jsonl", make_customers(n=3), now - timedelta(1))

    df = load_predictions(tmp_path, window_days=7, now=now)

    assert len(df) == 3


def test_load_predictions_handles_missing_directory(tmp_path: Path) -> None:
    assert load_predictions(tmp_path / "nope", window_days=7).empty
