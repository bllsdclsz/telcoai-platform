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
- `tune.py`: Optuna search (TPE, seeded) scored by 5-fold CV on the training split only, so the test metrics stay unbiased. Every trial is a nested MLflow run. `churn pipeline --tune-trials 30`
- `model_card.py`: model card generated for every registered version (Markdown + JSON in MLflow, tags and description on the version). See [Governance](#governance-model-card-and-prod-approval).
- `registry.py`: alias-based promotion with automatic `<env>-previous` for rollback.
- `serve/app.py`: FastAPI service that serves `models:/telco-churn@prod` and validates every request field. It follows the alias: a promotion or rollback takes effect within `CHURN_MODEL_REFRESH_SECONDS` (30 s) without a restart. `/health` is readiness (503 until a model is loaded), `/livez` is liveness.

## Feature store (Feast)

```
validated data (DVC) ──▶ offline store (parquet) ──materialize──▶ online store (Redis)
        │                         │                                     │
        ▼                         ▼                                     ▼
   model training          get_historical_features        feature server ◀── API /predict/by-id
```

- Definitions (`churn/feature_store.py`, registered by `churn/feature_repo/`): entity `customer`, feature view `customer_features` (19 features), and feature service `churn_model`, which the API requests by name.
- **`/predict/by-id`**: the call center sends only customer IDs. The API fetches the features from the Feast feature server over HTTP, which keeps the Feast SDK out of the serving image. The features are validated against the same `CustomerFeatures` contract as `/predict`, then scored. Unknown IDs return 404; a feature server that is down returns 503.
- **Offline/online parity:** tests check that the offline retrieval, the online retrieval and the source data are identical, and that the model gives bit-identical scores on both paths. Live check: `/predict/by-id` and `/predict` with the raw CSV record return exactly the same probability.

```bash
make up features serve-docker
curl -X POST localhost:8000/predict/by-id -H 'Content-Type: application/json' -d '{"customer_ids": ["7590-VHVEG"]}'
```

## Governance: model card and prod approval

Every registered version gets `model_card/model_card.md` and `.json` in its MLflow run:
- **Intended and out-of-scope use.**
- **Data lineage:** source and DVC md5.
- **Test metrics.**
- **Per-group performance:** churn rate, predicted rate, recall, FPR and ROC AUC per group.
- **Fairness gaps** for sensitive attributes: the recall gap (equal opportunity) and the predicted-rate gap (demographic parity), with a 10% tolerance.
- **Automatically detected blind spots,** plus limitations and privacy notes.

Promotion to `prod` is refused unless all three checks pass:
1. the version has a model card,
2. a named person approves it (`--approved-by`),
3. if the card flags a fairness gap, the reviewer acknowledges it (`--fairness-reviewed`).

The approver, a UTC timestamp and the review status are written onto the model version as tags, which gives an audit trail per prod model.

What the real model card showed:
- **Fairness:** gender gaps were small (≤ 2.7%). Senior citizens are flagged 18.9 points more often, which largely follows their real churn rate (44% vs 23%), and their recall gap was 6.2%, within tolerance. The version is flagged, and a reviewer has to acknowledge it.
- **Blind spots:** the model almost never flags one- or two-year contract customers, even though 12% of one-year customers churn.

## Baseline

| Metric   | Default params | Tuned (25 Optuna trials) |
| -------- | -------------- | ------------------------ |
| CV ROC AUC (train split) | 0.845 | 0.851 |
| Test ROC AUC  | 0.844 | 0.847 |
| Test PR AUC   | 0.655 | 0.665 |
| Test F1 @ 0.5 | 0.577 | 0.575 |

Tuning gives a small but real gain on the held-out test set. The data, not the hyperparameters, is the limiting factor.

## Usage

```bash
uv run churn download
uv run churn train                     # or: uv run churn pipeline (Prefect, promotes to staging)
uv run churn promote --from staging --to prod --approved-by "Your Name" [--fairness-reviewed]
uv run churn serve                     # http://127.0.0.1:8000/docs
```

All settings can be overridden with `CHURN_*` environment variables (see `config.py`).
Rollback: `uv run churn promote --from prod-previous --to prod --approved-by "Your Name"`.

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
make up train features               # MLflow + Redis, a trained model, online features
make promote APPROVED_BY="Your Name" FAIRNESS_REVIEWED=1
make serve-docker                    # API + feature server + Prometheus + Grafana
make demo-drift                      # normal traffic -> price-increase scenario -> drift check -> retrain to staging
```

Sample run: 6 of 19 features drifted (tenure, charges, contract, internet service, payment method), dataset drift 0.32 ≥ 0.25. Retraining registered a new version in `staging`, and `prod` was unchanged.
