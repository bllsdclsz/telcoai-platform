"""Runtime configuration, overridable through ``CHURN_*`` environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    raw_data_url: str = RAW_DATA_URL
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "telco-churn"
    registered_model_name: str = "telco-churn"
    # Alias assigned to freshly trained models; promotion to staging/prod is a separate step.
    register_alias: str = "dev"
    # Alias the serving API loads.
    serving_alias: str = "prod"
    random_seed: int = 42
    test_size: float = 0.2
    # Quality gate: a model below this ROC AUC is never registered.
    min_roc_auc: float = 0.80

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "telco_churn.csv"
