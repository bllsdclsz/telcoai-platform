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

## Monitoring & drift

```
API ──/metrics──▶ Prometheus (SLO rules + alerts) ──▶ Grafana "Churn API" dashboard
 │
 └─ prediction log (JSONL/day) ──▶ churn-drift-monitoring flow (Evidently)
                                     ├─ drift metrics + HTML report ──▶ MLflow "telco-churn-monitoring"
                                     └─ dataset drift? ──▶ churn-training flow ──▶ @staging (prod stays manual)
```

- **SLOs:** p95 `/predict` < 300 ms, 5xx < 1%, availability 99.5%. The alert rules are in `monitoring/prometheus/rules/`, and each alert links to its section of [the runbook](../docs/runbooks/churn-api.md).
- **Drift rule:** a feature drifts when its distance from the training distribution is ≥ 0.1 (Wasserstein for numeric features, Jensen-Shannon for categorical). The dataset drifts when ≥ 25% of features drift, over ≥ 200 predictions from the last 7 days.
- **Fast signal:** the `ChurnPredictionShift` alert fires when the predicted churn rate stays far above the 26.5% base rate. Evidently then shows *which* features moved.

Demo (full stack in Docker; Grafana at http://localhost:3000, Prometheus at :9090):

```bash
make up train promote serve-docker   # MLflow, model in prod, API + Prometheus + Grafana
make demo-drift                      # normal traffic -> price-increase scenario -> drift check -> retrain to staging
```

Sample run: 6 of 19 features drifted (tenure, charges, contract, internet service, payment method), dataset drift 0.32 ≥ 0.25. Retraining registered a new version in `staging`, and `prod` was unchanged.
