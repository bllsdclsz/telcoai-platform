"""Feast feature definitions and helpers: one set of customer features, served offline and online.

The offline store (parquet) is exported from the same validated, DVC-versioned dataset the model
is trained on; ``materialize`` copies the latest values to the online store (Redis in Docker,
SQLite in tests). Parity tests check that both paths return identical features.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

from churn.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SERVICE,
    ID_COLUMN,
    NUMERIC_COLUMNS,
)

TIMESTAMP_COLUMN = "event_timestamp"
# The public dataset is a single snapshot without timestamps; all rows get this one.
SNAPSHOT_TIME = datetime(2026, 9, 1, tzinfo=UTC)

customer = Entity(
    name="customer",
    join_keys=[ID_COLUMN],
    value_type=ValueType.STRING,
    description="Telco customer",
)

_SCHEMA = [
    *(Field(name=c, dtype=String) for c in CATEGORICAL_COLUMNS),
    Field(name="SeniorCitizen", dtype=Int64),
    Field(name="tenure", dtype=Int64),
    Field(name="MonthlyCharges", dtype=Float64),
    Field(name="TotalCharges", dtype=Float64),
]
assert {f.name for f in _SCHEMA} == set(FEATURE_COLUMNS)
assert set(NUMERIC_COLUMNS) <= {f.name for f in _SCHEMA}


def definitions(source_path: Path) -> list[Entity | FeatureView | FeatureService]:
    """Feast objects for ``feast apply``, reading the offline store at ``source_path``."""
    source = FileSource(
        name="customer_features_source",
        path=str(source_path.resolve()),
        timestamp_field=TIMESTAMP_COLUMN,
    )
    view = FeatureView(
        name="customer_features",
        entities=[customer],
        ttl=timedelta(days=365 * 5),
        schema=_SCHEMA,
        source=source,
        online=True,
        description="Customer profile, contract, services and billing (one row per customer)",
    )
    service = FeatureService(name=FEATURE_SERVICE, features=[view])
    return [customer, view, service]


def export_offline(
    customers: pd.DataFrame, path: Path, timestamp: datetime = SNAPSHOT_TIME
) -> Path:
    """Write validated customers as the offline store's parquet source."""
    out = customers[[ID_COLUMN, *FEATURE_COLUMNS]].copy()
    out[TIMESTAMP_COLUMN] = timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def publish(store: FeatureStore, customers: pd.DataFrame, source_path: Path) -> None:
    """Export -> register definitions -> materialize to the online store."""
    export_offline(customers, source_path)
    objects: list[Any] = definitions(source_path)  # Feast's apply() type is invariant
    store.apply(objects)
    store.materialize(start_date=SNAPSHOT_TIME - timedelta(days=1), end_date=datetime.now(UTC))


def offline_features(store: FeatureStore, customer_ids: list[str]) -> pd.DataFrame:
    """Point-in-time correct features, as a training job would retrieve them."""
    entity_df = pd.DataFrame({ID_COLUMN: customer_ids, TIMESTAMP_COLUMN: datetime.now(UTC)})
    df = store.get_historical_features(
        entity_df=entity_df, features=store.get_feature_service(FEATURE_SERVICE)
    ).to_df()
    return df.set_index(ID_COLUMN).loc[customer_ids, FEATURE_COLUMNS]


def online_features(store: FeatureStore, customer_ids: list[str]) -> pd.DataFrame:
    """Latest features from the online store, as the serving path retrieves them."""
    data = store.get_online_features(
        features=store.get_feature_service(FEATURE_SERVICE),
        entity_rows=[{ID_COLUMN: cid} for cid in customer_ids],
    ).to_dict()
    return pd.DataFrame(data).set_index(ID_COLUMN).loc[customer_ids, FEATURE_COLUMNS]
