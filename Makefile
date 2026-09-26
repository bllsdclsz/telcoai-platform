.PHONY: install lint test check mlflow download train promote serve up serve-docker down

export MLFLOW_DISABLE_AGENT_HINT := 1
COMPOSE := sh scripts/compose.sh

install:            ## create .venv with all packages and dev tools
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy churn/src

test:
	uv run pytest -p no:warnings

check: lint test    ## everything CI runs

mlflow:             ## local MLflow server without Docker (http://127.0.0.1:5000)
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

download:
	uv run churn download

train:
	uv run churn train

promote:            ## make promote FROM=dev TO=prod
	uv run churn promote --from $(or $(FROM),dev) --to $(or $(TO),prod)

serve:
	uv run churn serve

up:                 ## MLflow in Docker
	$(COMPOSE) up -d mlflow

serve-docker:       ## churn API in Docker (needs a model behind the prod alias)
	$(COMPOSE) --profile serve up -d --build churn-api

down:
	$(COMPOSE) --profile serve down
