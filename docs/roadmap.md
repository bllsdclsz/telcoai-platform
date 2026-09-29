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

| Capability                                                                                                                                                        | Status | Where                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| Fictional multilingual help center (21 topics × DE/FR/IT/EN), validated on load                                                                                   | Done   | `corpus/`, `corpus.py`                                                                               |
| Paragraph chunking, ONNX multilingual embeddings, Qdrant (embedded or server), language filter                                                                    | Done   | `chunking.py`, `embeddings.py`, `index.py`, `retrieve.py`                                            |
| Retrieval evaluation (94 golden questions), embedding-model benchmark, CI gate with thresholds                                                                    | Done   | `evaluate.py`, `eval/`, `.github/workflows/support-rag-ci.yml`                                       |
| Generation: LiteLLM provider layer (local Ollama), versioned prompts, cited answers, calibrated scope filter + model NO_ANSWER fallback, FastAPI `/ask`           | Done   | `llm.py`, `prompts/`, `assistant.py`, `api.py`                                                       |
| Guardrails: PII redaction, prompt-injection rules + learned classifier, prompt-leak canary, citation/number grounding; safety eval with held-out attacks, CI gate | Done   | `guardrails.py`, `injection_model.py`, `safety_eval.py`, `eval/safety.yaml`                          |
| Human approval for actions: tool-calling proposal (goodwill credit), validation + duration cross-check, pending queue, agent API, event log, action eval gate     | Done   | `actions.py`, `api.py`, `eval/actions.yaml`                                                          |
| Tracing + audit log: one MLflow trace per question (span per step), audit tags, PII only as types, `rag audit` queries, ~50 ms overhead                           | Done   | `tracing.py`, `assistant.py`                                                                         |
| Answer-quality evaluation: facts, style (language, repetition, echo), declines, calibrated LLM judge (faithful, relevant), MLflow runs, local gate                | Done   | `generation_eval.py`, `eval/generation_golden.yaml`, `eval/judge_calibration.yaml`, `prompts/judge/` |
| Offline A/B of models and prompt versions (e.g. `answer@v1` vs `answer@v2`) as MLflow runs                                                                        | Done   | `rag eval-generation --model … --prompt-version …`                                                   |

## Project 3: `platform/` (IaC, GitOps, enablement)

| Item                                                                                                   | Status  | Where                                                         |
| ------------------------------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------- |
| Separate k3d cluster (the Rancher Desktop cluster belongs to another project), isolated kubeconfig     | Done    | `platform/k3d/`, `platform/scripts/k3d.sh`, `make cluster-up` |
| Terraform: dev/test/prod namespaces with quotas and container defaults, Argo CD, app-of-apps root      | Done    | `platform/terraform/`                                         |
| Helm charts: shared MLflow registry, churn API serving one alias per environment; dev bootstraps its own model | Done | `platform/charts/` |
| Argo CD deploys dev from `main`; the churn image is published to GHCR on every merge                   | Done    | `platform/gitops/`, `.github/workflows/churn-ci.yml`          |
| Platform CI: helm lint, kubeconform schema checks, terraform fmt/validate                              | Done    | `.github/workflows/platform-ci.yml`                           |
| Test and prod: ApplicationSet (one app per env folder), dev/test/prod serve @dev/@staging/@prod of one registry | Done | `platform/gitops/` |
| Image promotion by PR (immutable `sha-` tags, prod only after test, checked in CI); rollback = revert | Done | `make promote-image`, `platform/scripts/check-promotions.sh` |
| API starts unready and polls the registry instead of exiting when its alias has no model | Next | `churn/src/churn/serve/` |
| Support assistant chart (Qdrant, Ollama)                                                               | Planned |                                                               |
| Copier service template                                                                                | Planned |                                                               |
| SLO report, runbooks, a postmortem                                                                     | Planned |                                                               |
| Optional one-weekend Azure run                                                                         | Planned |                                                               |

## Local environment notes

- **Docker:** runs on Rancher Desktop, shared with another project. `scripts/compose.sh` falls back to `rdctl shell` when the Windows Docker pipe is unavailable.
- **Ports used:** 5000 (MLflow), 6379 (Redis), 6566 (Feast), 8000 (API), 9090 (Prometheus), 3000 (Grafana); k3d: 6550 (Kubernetes API), 8080/8443 (ingress), 8081 (Argo CD UI port-forward).
- **Kubernetes:** k3d runs inside the Rancher Desktop VM when the Windows Docker pipe is unavailable (`platform/scripts/k3d.sh`). All commands use `platform/.kube/telcoai.yaml`; the current kubectl context is never changed.
- **Resuming after a restart:** `make up data train promote features serve-docker`. Volumes keep MLflow runs and the registry, so `train promote` is only needed on a fresh volume.
