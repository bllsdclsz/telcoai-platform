# Runbook: Churn API

Service: `churn-api` (FastAPI, serves `models:/telco-churn@prod`)
Dashboard: Grafana → TelcoAI → **Churn API** · Metrics: `http://<host>:8000/metrics` · Alert rules: `monitoring/prometheus/rules/churn-api.yml`

| SLI                        | SLO      |
| -------------------------- | -------- |
| Availability (scrape `up`) | 99.5%    |
| p95 `/predict` latency     | < 300 ms |
| 5xx ratio on `/predict`    | < 1%     |

## ChurnApiDown

**Meaning:** Prometheus could not scrape the API for 1 minute.

1. Check the container: `make ps`, then `sh scripts/compose.sh logs --tail 100 churn-api`.
2. **Startup failure** ("RESOURCE_DOES_NOT_EXIST" / alias not found): no model sits behind the serving alias. Check the registry (MLflow UI → Models → telco-churn → aliases), then run `uv run churn promote --from staging --to prod --approved-by "<you>"` (add `--fairness-reviewed` if its model card flags a gap).
3. **MLflow unreachable:** the API loads the model at startup, so it needs MLflow up (`curl localhost:5000/health`).
4. Restart: `make serve-docker`.

## ChurnApiHighLatency

**Meaning:** p95 `/predict` latency has been above 300 ms for 5 minutes.

1. Look at the **Requests / s** panel. Is it a traffic spike? Batch callers should use batches of 1,000 customers or fewer (the request limit).
2. Look at **Model version**. Did a promotion just happen? A heavier model can be slower. If so, roll back with `uv run churn promote --from prod-previous --to prod --approved-by "<you>"`, then restart the API. The rollback target was already approved, so no new fairness review is needed.
3. Check CPU on the host (`docker stats`). Scale out by running more replicas behind a load balancer (Kubernetes deployment, Project 3).

## ChurnApiHighErrorRate

**Meaning:** More than 1% of `/predict` calls returned 5xx for 5 minutes. A 422 (invalid input) is a client error and does **not** count.

1. Check the logs for the traceback: `sh scripts/compose.sh logs --tail 200 churn-api`.
2. If the errors started after a promotion, roll back (see above).
3. **Only `/predict/by-id` fails with 503:** the Feast feature server or Redis is down (`make ps`). `/predict` with full records keeps working. Restart with `make serve-docker`. If Redis lost its data, republish with `make features`.
4. If they come from specific inputs, capture a failing payload, add it as a test, and fix it.

## ChurnPredictionShift

**Meaning:** Over the last hour, more than 45% of scored customers were predicted to churn (training base rate: 26.5%), across at least 200 predictions.

The input population has probably changed (a campaign, a price change, a new sales channel) or an upstream system is sending malformed data.

1. Run the drift check: `uv run churn monitor --no-retrain`. The flow logs per-feature drift scores and an HTML report to the MLflow experiment `telco-churn-monitoring`.
2. **If the drifted features match a known business change:** let the monitoring flow retrain (`uv run churn monitor`). The candidate goes to `staging`, where you compare its metrics with `prod` before promoting it manually.
3. **If they don't match:** suspect the data. Check the latest `data/predictions/*.jsonl` for impossible values and contact the calling team.
4. Record the outcome as a postmortem if customers were affected.
