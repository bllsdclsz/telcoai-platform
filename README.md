# TelcoAI Platform

An end-to-end ML and GenAI platform for a telecom operator: it covers taking models from experimentation to production, with the engineering standards a central ML platform team would provide to product teams.

| Component          | What it shows                                                                                                                             | Status      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [`churn/`](churn/) | Churn prediction: validated data, reproducible training, MLflow registry with dev→staging→prod aliases, quality gate, FastAPI serving, CI | In progress |
| `support-rag/`     | Multilingual (DE/FR/IT/EN) customer-support RAG assistant: evaluation harness, guardrails, audit logging                                  | Planned     |
| `platform/`        | Terraform, Kubernetes, Argo CD GitOps, monitoring/SLOs, service template, runbooks                                                        | Planned     |

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Docker is optional.

```bash
make install          # .venv + pre-commit hooks
make mlflow           # MLflow at http://127.0.0.1:5000 (or `make up` for the Docker version)
make download train   # fetch data, train, register as telco-churn@dev if it passes the gate
make promote FROM=dev TO=prod
make serve            # API at http://127.0.0.1:8000/docs
```

Full stack in Docker (MLflow, Redis + Feast feature server, API, Prometheus, Grafana):

```bash
make up data train promote features serve-docker
make demo-drift       # simulated drift -> automatic retrain to staging
```

## Engineering standards

- **Reproducible:** Python and every dependency pinned in `uv.lock`; seeded training; data and parameters logged per run.
- **Quality gates:** schema validation (pandera) before training; a model below the metric threshold is never registered.
- **Promotion by alias:** training only ever writes `dev`. Promotion is an explicit step, and the previous version is kept as `<env>-previous` for one-command rollback.
- **CI on every change:** ruff, mypy, pytest, then a container build.

Progress and next steps: [docs/roadmap.md](docs/roadmap.md). Tool versions: [docs/versions.md](docs/versions.md). Runbooks: [docs/runbooks/](docs/runbooks/).
