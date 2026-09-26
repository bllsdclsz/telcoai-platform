from pathlib import Path

import pandas as pd
import pytest
from feast import FeatureStore
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RegistryConfig, RepoConfig

from churn.feature_store import offline_features, online_features, publish
from churn.schema import FEATURE_COLUMNS, ID_COLUMN
from churn.train import build_pipeline

from .conftest import make_customers


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return make_customers(n=200, seed=3)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory, customers: pd.DataFrame) -> FeatureStore:
    """Same definitions as churn/feature_repo, with SQLite instead of Redis for the online store."""
    root = tmp_path_factory.mktemp("feast")
    config = RepoConfig(
        project="churn_test",
        provider="local",
        registry=RegistryConfig(path=str(root / "registry.db")),
        online_store=SqliteOnlineStoreConfig(path=str(root / "online.db")),
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
        repo_path=root,
    )
    fs = FeatureStore(config=config)
    publish(fs, customers, root / "customer_features.parquet")
    return fs


def source_rows(customers: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return customers.set_index(ID_COLUMN).loc[ids, FEATURE_COLUMNS]


def test_online_features_match_source_data(store: FeatureStore, customers: pd.DataFrame) -> None:
    ids = customers[ID_COLUMN].sample(25, random_state=0).tolist()
    pd.testing.assert_frame_equal(
        online_features(store, ids), source_rows(customers, ids), check_dtype=False
    )


def test_offline_and_online_features_are_identical(
    store: FeatureStore, customers: pd.DataFrame
) -> None:
    """Offline/online parity: what training retrieves is exactly what serving retrieves."""
    ids = customers[ID_COLUMN].sample(25, random_state=1).tolist()
    pd.testing.assert_frame_equal(
        offline_features(store, ids), online_features(store, ids), check_dtype=False
    )


def test_model_scores_identically_on_both_paths(
    store: FeatureStore, customers: pd.DataFrame
) -> None:
    model = build_pipeline({"n_estimators": 20}, seed=0).fit(
        customers[FEATURE_COLUMNS], customers["Churn"]
    )
    ids = customers[ID_COLUMN].head(30).tolist()
    offline = model.predict_proba(offline_features(store, ids))[:, 1]
    online = model.predict_proba(online_features(store, ids))[:, 1]
    assert (offline == online).all()


def test_unknown_customer_has_no_online_features(store: FeatureStore) -> None:
    assert online_features(store, ["NOPE-0000"]).isna().all(axis=None)


def test_feature_repo_definitions_import() -> None:
    import importlib.util

    path = Path(__file__).parents[1] / "feature_repo" / "definitions.py"
    spec = importlib.util.spec_from_file_location("definitions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.churn_model.name == "churn_model"
    assert {f.name for f in module.customer_features.schema} >= set(FEATURE_COLUMNS)
