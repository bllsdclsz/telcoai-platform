# Roadmap

Status as of 2026-09-26.

## Project 1: `churn/` (Churn prediction MLOps pipeline)

| Capability                                                                                      | Status  | Where                                                    |
| ----------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| Validated data (pandera), DVC versioning, data hash on every run                                | Done    | `schema.py`, `data.py`, `data/raw/*.dvc`                 |
| Reproducible training, MLflow tracking, quality gate                                            | Done    | `train.py`                                               |
| Registry promotion dev → staging → prod, one-command rollback                                   | Done    | `registry.py`                                            |
| Real-time API with input contract                                                               | Done    | `serve/app.py`                                           |
| Prefect training flow (prod promotion stays manual)                                             | Done    | `flows.py`                                               |
| Docker image (non-root), compose stack                                                          | Done    | `churn/Dockerfile`, `docker-compose.yml`                 |
| CI: lint, types, tests, image build, rules/dashboard validation; SHA-pinned actions; Dependabot | Done    | `.github/`                                               |
| Prometheus metrics, SLO alert rules, Grafana dashboard, runbook                                 | Done    | `monitoring/`, `docs/runbooks/`                          |
| Evidently drift detection → automatic retrain to staging                                        | Done    | `drift.py`, `flows.py`                                   |
| Feast feature store (offline parquet, online Redis), `/predict/by-id`, parity tests             | Done    | `feature_store.py`, `feature_repo/`, `serve/features.py` |
| **Model card** generated per registered version (data, metrics, fairness slices, limitations)   | Next    |                                                          |
| **Optuna tuning** in the training flow                                                          | Next    |                                                          |
| Batch scoring job (nightly retention list)                                                      | Planned |                                                          |
| Slimmer serving image (currently 944 MB, mostly MLflow)                                         | Planned |                                                          |

## Project 2: `support-rag/` (Multilingual support assistant, DE/FR/IT/EN)

Planned: provider-agnostic LLM layer (LiteLLM, plus Ollama for free local runs and CI), multilingual embeddings with hybrid search, versioned prompts, an evaluation harness (golden set, faithfulness, safety/red-team, per-language) used as a CI gate, guardrails (grounding, PII redaction, human approval for actions), audit logging and tracing, and A/B testing of prompt versions.

## Project 3: `platform/` (IaC, GitOps, enablement)

Planned:

- **Cluster:** a separate k3d cluster (the Rancher Desktop cluster belongs to another project), with Terraform for cluster add-ons.
- **GitOps:** Argo CD with dev/test/prod overlays.
- **Enablement:** a copier service template.
- **SLOs:** SLO report and a postmortem.
- **Cloud:** an optional one-weekend Azure run.

## Local environment notes

- **Docker:** runs on Rancher Desktop, shared with another project. `scripts/compose.sh` falls back to `rdctl shell` when the Windows Docker pipe is unavailable.
- **Ports used:** 5000 (MLflow), 6379 (Redis), 6566 (Feast), 8000 (API), 9090 (Prometheus), 3000 (Grafana).
- **Resuming after a restart:** `make up data train promote features serve-docker`. Volumes keep MLflow runs and the registry, so `train promote` is only needed on a fresh volume.
