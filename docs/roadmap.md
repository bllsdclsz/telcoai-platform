# Roadmap

Status as of 2026-09-29.

## Project 1: `churn/` (Churn prediction MLOps pipeline)

| Capability                                                                                               | Status  | Where                                                    |
| -------------------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| Validated data (pandera), DVC versioning, data hash on every run                                         | Done    | `schema.py`, `data.py`, `data/raw/*.dvc`                 |
| Reproducible training, MLflow tracking, quality gate                                                     | Done    | `train.py`                                               |
| Registry promotion dev → staging → prod, one-command rollback                                            | Done    | `registry.py`                                            |
| Real-time API with input contract                                                                        | Done    | `serve/app.py`                                           |
| Prefect training flow (prod promotion stays manual)                                                      | Done    | `flows.py`                                               |
| Docker image (non-root), compose stack                                                                   | Done    | `churn/Dockerfile`, `docker-compose.yml`                 |
| CI: lint, types, tests, image build, rules/dashboard validation; SHA-pinned actions; Dependabot          | Done    | `.github/`                                               |
| Prometheus metrics, SLO alert rules, Grafana dashboard, runbook                                          | Done    | `monitoring/`, `docs/runbooks/`                          |
| Evidently drift detection → automatic retrain to staging                                                 | Done    | `drift.py`, `flows.py`                                   |
| Feast feature store (offline parquet, online Redis), `/predict/by-id`, parity tests                      | Done    | `feature_store.py`, `feature_repo/`, `serve/features.py` |
| Model card per registered version (intended use, lineage, per-group metrics, fairness gaps, blind spots) | Done    | `model_card.py`                                          |
| Prod approval gate: model card + named approver + fairness acknowledgement, recorded on the version      | Done    | `registry.py`                                            |
| Optuna tuning (CV on the training split, nested MLflow runs)                                             | Done    | `tune.py`                                                |
| Batch scoring job (nightly retention list)                                                               | Planned |                                                          |
| Slimmer serving image (currently 944 MB, mostly MLflow)                                                  | Planned |                                                          |

## Project 2: `support-rag/` (Multilingual support assistant, DE/FR/IT/EN)

| Capability                                                                                                                                                        | Status  | Where                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------- |
| Fictional multilingual help center (21 topics × DE/FR/IT/EN), validated on load                                                                                   | Done    | `corpus/`, `corpus.py`                                                      |
| Paragraph chunking, ONNX multilingual embeddings, Qdrant (embedded or server), language filter                                                                    | Done    | `chunking.py`, `embeddings.py`, `index.py`, `retrieve.py`                   |
| Retrieval evaluation (94 golden questions), embedding-model benchmark, CI gate with thresholds                                                                    | Done    | `evaluate.py`, `eval/`, `.github/workflows/support-rag-ci.yml`              |
| Generation: LiteLLM provider layer (local Ollama), versioned prompts, cited answers, calibrated scope filter + model NO_ANSWER fallback, FastAPI `/ask`           | Done    | `llm.py`, `prompts/`, `assistant.py`, `api.py`                              |
| Guardrails: PII redaction, prompt-injection rules + learned classifier, prompt-leak canary, citation/number grounding; safety eval with held-out attacks, CI gate | Done    | `guardrails.py`, `injection_model.py`, `safety_eval.py`, `eval/safety.yaml` |
| **Human approval for actions** (e.g. goodwill credit proposed by the assistant, approved by an agent)                                                             | Planned |                                                                             |
| **Audit & tracing:** MLflow tracing of every request (prompt version, sources, latency)                                                                           | Planned |                                                                             |
| **Generation evaluation:** faithfulness and correctness (LLM judge), safety/red-team set, answer-language check, CI gate                                          | Planned |                                                                             |
| A/B testing of prompt versions                                                                                                                                    | Planned |                                                                             |

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
