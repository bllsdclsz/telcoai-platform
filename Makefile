.PHONY: install lint test check mlflow download data train features pipeline promote serve monitor simulate rag-ingest rag-eval demo-drift up serve-docker ps down

export MLFLOW_DISABLE_AGENT_HINT := 1
COMPOSE := sh scripts/compose.sh

install:            ## create .venv with all packages and dev tools
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy churn/src
	uv run mypy support-rag/src

test:
	uv run pytest -p no:warnings

check: lint test    ## everything CI runs

mlflow:             ## local MLflow server without Docker (http://127.0.0.1:5000)
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

download:
	uv run churn download

train:
	uv run churn train

features:           ## publish customer features to Feast (needs Redis: make up)
	uv run churn features

pipeline:           ## Prefect flow: fetch -> [tune] -> train + gate + model card -> staging (TUNE=30)
	uv run churn pipeline $(if $(TUNE),--tune-trials $(TUNE))

data:               ## pull the versioned dataset (falls back to download)
	uv run dvc pull || uv run churn download

promote:            ## make promote FROM=staging TO=prod APPROVED_BY="Name" [FAIRNESS_REVIEWED=1]
	uv run churn promote --from $(or $(FROM),dev) --to $(or $(TO),prod) \
		$(if $(APPROVED_BY),--approved-by "$(APPROVED_BY)") $(if $(FAIRNESS_REVIEWED),--fairness-reviewed)

serve:
	uv run churn serve

rag-ingest:         ## index the help-center corpus (support-rag)
	uv run rag ingest

rag-eval:           ## retrieval + safety evaluation gates (support-rag)
	uv run rag eval-retrieval --gate
	uv run rag eval-safety --gate

monitor:            ## Prefect drift check on recent predictions; retrains on drift
	uv run churn monitor

simulate:           ## normal traffic to the API (add DRIFT=1 for the shift scenario)
	uv run churn simulate --n $(or $(N),300) $(if $(DRIFT),--drift)

demo-drift:         ## normal traffic -> drifted traffic -> drift check -> retrain to staging
	uv run churn simulate --n 300
	uv run churn simulate --n 600 --drift --seed 1
	uv run churn monitor

up:                 ## MLflow + Redis (online feature store) in Docker
	$(COMPOSE) up -d mlflow redis

serve-docker:       ## API + Prometheus (:9090) + Grafana (:3000) in Docker; needs a prod model
	$(COMPOSE) --profile serve up -d --build

ps:
	$(COMPOSE) --profile serve ps

down:
	$(COMPOSE) --profile serve down
