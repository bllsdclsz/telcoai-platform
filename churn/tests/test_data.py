from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from churn.data import clean, load


def test_clean_fills_blank_total_charges_for_new_customers(raw_csv: Path) -> None:
    df = clean(pd.read_csv(raw_csv))
    assert df["TotalCharges"].dtype == float
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}


def test_load_returns_validated_frame(raw_csv: Path) -> None:
    df = load(raw_csv)
    assert len(df) == 400
    assert df["customerID"].is_unique


def test_load_rejects_unknown_category(raw_csv: Path) -> None:
    raw = pd.read_csv(raw_csv)
    raw.loc[0, "Contract"] = "Lifetime"
    raw.to_csv(raw_csv, index=False)
    with pytest.raises(pandera.errors.SchemaErrors):
        load(raw_csv)


def test_load_rejects_negative_charges(raw_csv: Path) -> None:
    raw = pd.read_csv(raw_csv)
    raw.loc[0, "MonthlyCharges"] = -5
    raw.to_csv(raw_csv, index=False)
    with pytest.raises(pandera.errors.SchemaErrors):
        load(raw_csv)
