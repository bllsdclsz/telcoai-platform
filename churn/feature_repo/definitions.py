"""Entry point for `feast apply`: the definitions live in the churn package (shared with tests)."""

from pathlib import Path

from churn.feature_store import definitions

OFFLINE_SOURCE = Path(__file__).parents[2] / "data" / "feast" / "customer_features.parquet"

customer, customer_features, churn_model = definitions(OFFLINE_SOURCE)
