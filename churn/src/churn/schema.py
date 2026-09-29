"""Single source of truth for the customer record: category sets, pandera schema."""

from typing import Literal, get_args

import pandera.pandas as pa
from pandera.typing import Series

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
MultipleLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
Gender = Literal["Male", "Female"]

ID_COLUMN = "customerID"
TARGET = "Churn"
# Feast feature service the model reads (see churn.feature_store).
FEATURE_SERVICE = "churn_model"

CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "gender": get_args(Gender),
    "Partner": get_args(YesNo),
    "Dependents": get_args(YesNo),
    "PhoneService": get_args(YesNo),
    "MultipleLines": get_args(MultipleLines),
    "InternetService": get_args(InternetService),
    "OnlineSecurity": get_args(InternetAddon),
    "OnlineBackup": get_args(InternetAddon),
    "DeviceProtection": get_args(InternetAddon),
    "TechSupport": get_args(InternetAddon),
    "StreamingTV": get_args(InternetAddon),
    "StreamingMovies": get_args(InternetAddon),
    "Contract": get_args(Contract),
    "PaperlessBilling": get_args(YesNo),
    "PaymentMethod": get_args(PaymentMethod),
}
CATEGORICAL_COLUMNS = list(CATEGORY_VALUES)
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def _isin(column: str) -> pa.Field:  # type: ignore[valid-type]
    return pa.Field(isin=list(CATEGORY_VALUES[column]))


class CustomerSchema(pa.DataFrameModel):
    """Cleaned customer record, as produced by :func:`churn.data.clean`."""

    customerID: Series[str] = pa.Field(unique=True)
    gender: Series[str] = _isin("gender")
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = _isin("Partner")
    Dependents: Series[str] = _isin("Dependents")
    tenure: Series[int] = pa.Field(ge=0, le=120)
    PhoneService: Series[str] = _isin("PhoneService")
    MultipleLines: Series[str] = _isin("MultipleLines")
    InternetService: Series[str] = _isin("InternetService")
    OnlineSecurity: Series[str] = _isin("OnlineSecurity")
    OnlineBackup: Series[str] = _isin("OnlineBackup")
    DeviceProtection: Series[str] = _isin("DeviceProtection")
    TechSupport: Series[str] = _isin("TechSupport")
    StreamingTV: Series[str] = _isin("StreamingTV")
    StreamingMovies: Series[str] = _isin("StreamingMovies")
    Contract: Series[str] = _isin("Contract")
    PaperlessBilling: Series[str] = _isin("PaperlessBilling")
    PaymentMethod: Series[str] = _isin("PaymentMethod")
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0)
    Churn: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True
