"""Data drift between training data and recently scored customers, using Evidently."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from churn.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

# Fixed distance metrics (instead of Evidently's sample-size-dependent defaults, which switch
# between distances and p-values) so "drifted" always means "distance >= threshold".
NUMERIC_METHOD = "wasserstein"
CATEGORICAL_METHOD = "jensenshannon"


@dataclass(frozen=True)
class DriftResult:
    n_reference: int
    n_current: int
    column_scores: dict[str, float]
    threshold: float
    drift_share_threshold: float
    drifted_columns: list[str] = field(default_factory=list)

    @property
    def drift_share(self) -> float:
        return len(self.drifted_columns) / len(self.column_scores)

    @property
    def dataset_drift(self) -> bool:
        return self.drift_share >= self.drift_share_threshold


def load_predictions(log_dir: Path, window_days: int, now: datetime | None = None) -> pd.DataFrame:
    """Scored customers from the last ``window_days`` days of prediction logs."""
    now = now or datetime.now(UTC)
    since = now - timedelta(days=window_days)
    files = sorted(log_dir.glob("predictions-*.jsonl")) if log_dir.exists() else []
    frames = [pd.read_json(f, lines=True, dtype={"model_version": str}) for f in files]
    if not frames:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "scored_at"])
    df = pd.concat(frames, ignore_index=True)
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    return df[df["scored_at"] >= since].reset_index(drop=True)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.1,
    drift_share: float = 0.25,
    html_report: Path | None = None,
) -> DriftResult:
    definition = DataDefinition(
        numerical_columns=NUMERIC_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS
    )
    preset = DataDriftPreset(
        columns=FEATURE_COLUMNS,
        num_method=NUMERIC_METHOD,
        cat_method=CATEGORICAL_METHOD,
        threshold=threshold,
        drift_share=drift_share,
    )
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current[FEATURE_COLUMNS], data_definition=definition),
        Dataset.from_pandas(reference[FEATURE_COLUMNS], data_definition=definition),
    )
    if html_report:
        html_report.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_report))

    scores = {
        m["config"]["column"]: float(m["value"])
        for m in snapshot.dict()["metrics"]
        if m["config"]["type"].endswith(":ValueDrift")
    }
    return DriftResult(
        n_reference=len(reference),
        n_current=len(current),
        column_scores=scores,
        threshold=threshold,
        drift_share_threshold=drift_share,
        drifted_columns=sorted(c for c, s in scores.items() if s >= threshold),
    )
