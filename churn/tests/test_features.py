import pandas as pd

from churn.features import ENGINEERED_COLUMNS, FeatureEngineer, build_preprocessor
from churn.schema import CATEGORY_VALUES, FEATURE_COLUMNS, NUMERIC_COLUMNS


def test_feature_engineer_adds_business_features(customers: pd.DataFrame) -> None:
    out = FeatureEngineer().fit_transform(customers[FEATURE_COLUMNS])
    assert set(ENGINEERED_COLUMNS) <= set(out.columns)
    assert out["num_addons"].between(0, 6).all()
    new = out["tenure"] == 0
    assert (out.loc[new, "avg_monthly_spend"] == out.loc[new, "TotalCharges"]).all()


def test_feature_engineer_does_not_mutate_input(customers: pd.DataFrame) -> None:
    X = customers[FEATURE_COLUMNS]
    before = X.copy()
    FeatureEngineer().transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_preprocessor_has_fixed_width_and_ignores_unknown_categories(
    customers: pd.DataFrame,
) -> None:
    X = FeatureEngineer().transform(customers[FEATURE_COLUMNS])
    pre = build_preprocessor().fit(X)
    expected = (
        sum(len(v) for v in CATEGORY_VALUES.values())
        + len(NUMERIC_COLUMNS)
        + len(ENGINEERED_COLUMNS)
    )
    assert pre.transform(X).shape == (len(X), expected)

    unseen = X.head(1).copy()
    unseen["Contract"] = "Lifetime"
    assert pre.transform(unseen).shape == (1, expected)
