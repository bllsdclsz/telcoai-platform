.PHONY: install lint test check mlflow download data train features pipeline promote serve monitor simulate rag-ingest rag-eval rag-eval-gen demo-drift up serve-docker ps down cluster-up cluster-down kubeconfig platform-apply argocd-ui registry-ui promote-image new-service

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

rag-eval-gen:       ## answer-quality gate with local LLMs + judge (before prompt/model changes)
	uv run rag calibrate-judge
	uv run rag eval-generation --gate
	uv run rag eval-actions --gate

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

# Project 3: local Kubernetes platform. Everything uses the isolated kubeconfig below; the
# current kubectl context is never changed.
KUBE := KUBECONFIG=$(CURDIR)/platform/.kube/telcoai.yaml
K3D := sh platform/scripts/k3d.sh

cluster-up:         ## k3d cluster "telcoai" (API on :6550, ingress on :8080) + kubeconfig
	$(K3D) cluster create --config platform/k3d/cluster.yaml
	$(MAKE) kubeconfig

cluster-down:
	$(K3D) cluster delete telcoai

kubeconfig:         ## write platform/.kube/telcoai.yaml
	mkdir -p platform/.kube
	$(K3D) kubeconfig get telcoai | sed -E 's#server: https://[^:]+:[0-9]+#server: https://127.0.0.1:6550#' > platform/.kube/telcoai.yaml

platform-apply:     ## Terraform: namespaces, quotas, Argo CD, root app (Argo CD then syncs platform/gitops)
	cd platform/terraform && terraform init -input=false && terraform apply -auto-approve -input=false

argocd-ui:          ## Argo CD on http://localhost:8081 (user admin, password printed first)
	@$(KUBE) kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d; echo
	$(KUBE) kubectl -n argocd port-forward svc/argocd-server 8081:80

registry-ui:        ## shared MLflow registry on http://localhost:5001 (then: make promote with CHURN_MLFLOW_TRACKING_URI=http://localhost:5001)
	$(KUBE) kubectl -n telcoai-registry port-forward svc/mlflow 5001:5000

promote-image:      ## PR that deploys an image: make promote-image [SERVICE=churn-api] ENV=test [SHA=<commit>] | ENV=prod
	sh platform/scripts/promote-image.sh $(or $(SERVICE),churn-api) $(ENV) $(SHA)

new-service:        ## scaffold a model service wired into the platform: make new-service NAME=fraud-score
	uvx copier@9.18.2 copy --trust --data service_name=$(NAME) --answers-file $(NAME)/.copier-answers.yml templates/ml-service .
