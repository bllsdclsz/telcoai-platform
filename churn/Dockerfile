# Build from the repo root:  docker build -f churn/Dockerfile -t telco-churn-api .
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.14.7-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies first so they are cached independently of source changes.
COPY pyproject.toml uv.lock ./
COPY churn/pyproject.toml churn/README.md churn/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package churn --no-install-workspace
COPY churn/src churn/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package churn --no-editable

FROM python:3.14.7-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "churn.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
