# churn: Telco customer churn prediction

Predicts which customers are likely to cancel, both for the call center (real-time API) and for nightly retention campaigns (batch; planned).

Data: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), 7,043 customers, 26.5% churn.
The dataset is versioned with DVC (`data/raw/telco_churn.csv.dvc`). Every MLflow run is tagged with the same `data_md5`, so any registered model can be traced back to the exact data it was trained on.

## Pipeline

```
download ─▶ clean + validate (pandera) ─▶ features ─▶ LightGBM ─▶ evaluate ─▶ quality gate ─▶ MLflow registry @dev
                                                                                                   │ churn promote
                                                                                                   ▼
                                                                              @staging ─▶ @prod ─▶ FastAPI /predict
```

- `schema.py`: one definition of the customer record, shared by data validation and the API request model, so training and serving can't drift apart.
- `features.py`: feature engineering as sklearn transformers, packaged inside the model artifact.
- `train.py`: stratified split, MLflow tracking (params, metrics, dataset lineage). The model is registered only if ROC AUC ≥ `CHURN_MIN_ROC_AUC` (default 0.80).
- `flows.py`: Prefect flow `churn-training` (fetch → train + gate → promote to `staging`) with retries on the data fetch. Automatic promotion to `prod` is refused on purpose, because prod needs a human approval.
- `registry.py`: alias-based promotion with automatic `<env>-previous` for rollback.
- `serve/app.py`: FastAPI service that loads `models:/telco-churn@prod` and validates every request field.

## Baseline

| Metric   | Test set (20%) |
| -------- | -------------- |
| ROC AUC  | 0.844          |
| PR AUC   | 0.655          |
| F1 @ 0.5 | 0.577          |

## Usage

```bash
uv run churn download
uv run churn train                     # or: uv run churn pipeline (Prefect, promotes to staging)
uv run churn promote --from staging --to prod
uv run churn serve                     # http://127.0.0.1:8000/docs
```

All settings can be overridden with `CHURN_*` environment variables (see `config.py`).
Rollback: `uv run churn promote --from prod-previous --to prod`.
