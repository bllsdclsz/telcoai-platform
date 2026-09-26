# Versions

Policy: use the latest stable release of every tool, locked in `uv.lock` (Python packages) and pinned by exact tag for container images and CI actions. Checked 2026-09-26.

| Component            | Version                                                   | Notes                                                                             |
| -------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Python               | 3.14.7                                                    | Newest stable version that all dependencies install on                            |
| uv                   | 0.12.19                                                   | Also the builder image `ghcr.io/astral-sh/uv:0.12.19`                             |
| MLflow               | 3.16.1                                                    | Registry **aliases** (stages are deprecated); models saved with skops, not pickle |
| scikit-learn         | 1.9.1                                                     |                                                                                   |
| LightGBM             | 4.7.0                                                     |                                                                                   |
| pandas               | 2.3.3                                                     | **Held back:** feast 0.66 requires `pandas<3` (latest is 3.0.6)                   |
| pandera              | 0.33.1                                                    | Uses the `pandera.pandas` import path                                             |
| FastAPI / Pydantic   | 0.141.1 / 2.13.5                                          |                                                                                   |
| Prefect              | 3.8.6                                                     | Pipeline extra                                                                    |
| Feast                | 0.66.0                                                    | Pipeline extra                                                                    |
| Evidently            | 0.7.23                                                    | Pipeline extra                                                                    |
| DVC                  | 3.67.1                                                    | Pipeline extra                                                                    |
| Optuna               | 5.0.0                                                     | Pipeline extra                                                                    |
| pytest / ruff / mypy | 9.1.1 / 0.16.9 / 2.3.1                                    |                                                                                   |
| prometheus-client | 0.24.1 | **Held back:** feast 0.66 requires `prometheus-client<0.25` (latest is 0.26.0) |
| httpx2               | 2.13.1                                                    | Starlette's test client has deprecated `httpx` in favour of `httpx2`              |
| MLflow server image  | `ghcr.io/mlflow/mlflow:v3.16.1`                           |                                                                                   |
| Prometheus / Grafana images | `prom/prometheus:v3.15.0` / `grafana/grafana:13.2.2` | |
| Redis / Feast feature server images | `redis:8.8.3-alpine` / `quay.io/feastdev/feature-server:0.66.0` | Feast images moved from Docker Hub (last: 0.46) to Quay |
| Init container image | `busybox:1.38.0` | Hands the prediction log bind mount to the API's non-root user |
| Base image           | `python:3.14.7-slim`                                      |                                                                                   |
| GitHub Actions       | checkout v7.0.1, setup-uv v10.2.0, setup-buildx v4.4.1, build-push v7.4.0 | Pinned by commit SHA; Dependabot opens update PRs weekly |                                                                                   |

## Upgrading

```bash
uv lock --upgrade && make check
```

Feast holds back two packages (pandas, prometheus-client). Re-check both caps whenever Feast releases a new version.
