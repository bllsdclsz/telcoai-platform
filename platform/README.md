# Platform: Kubernetes, Terraform, GitOps

The hub layer: product teams ship a Helm chart and an image; the platform gives them environments with a budget and deploys whatever `main` says. Nobody runs `kubectl apply` by hand.

```
GitHub (main) ── churn-ci: test → image → ghcr.io/…/telco-churn-api:{main, sha-<commit>}
      │
      ▼ polled by
Argo CD ── root app "platform-apps" (platform/gitops/apps/)
   ▲         ├─ mlflow-registry ─────────────► telcoai-registry   MLflow: one registry, aliases @dev @staging @prod
   │         └─ ApplicationSet churn-api (one app per folder in gitops/envs/)
   │              ├─ churn-api-dev  ─────────► telcoai-dev    image :main          serves @dev   (+ bootstrap training)
   │              ├─ churn-api-test ─────────► telcoai-test   image :sha-<commit>  serves @staging
   │              └─ churn-api-prod ─────────► telcoai-prod   image :sha-<commit>  serves @prod
Terraform ── namespaces with quotas + container defaults, Argo CD, root app
   ▲
k3d cluster "telcoai" (1 server + 1 agent)
```

## Two promotion tracks

Code and models move separately, and each move leaves a record.

| What moves           | How                                                                                                | Gate                                                                                     | Record                    | Rollback                                  |
| -------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------- |
| **Image** (API code) | `make promote-image ENV=test [SHA=…]`, then `ENV=prod`: opens a PR that edits one tag              | CI: immutable `sha-` tag, exists in GHCR, prod only runs an image that has been in test  | the merged PR             | revert the PR                             |
| **Model** (registry) | `make registry-ui`, then `make promote FROM=dev TO=staging` / `FROM=staging TO=prod APPROVED_BY=…` | quality gate at training; prod needs a named approver, a model card, the fairness review | tags on the model version | `make promote FROM=prod-previous TO=prod APPROVED_BY=…` |

Dev needs neither: it runs the latest image from `main` and the newest model that passed the gate.

## Run it

Needs k3d, Terraform, Helm and kubectl (`scoop install k3d terraform helm kubectl`) and a Docker engine (Rancher Desktop works).

```bash
make cluster-up       # k3d cluster + platform/.kube/telcoai.yaml
make platform-apply   # namespaces, quotas, Argo CD, root app
make argocd-ui        # prints the admin password, then http://localhost:8081
```

Argo CD then deploys the registry and the three environments on its own. In dev, the bootstrap job trains a model (a few minutes on a fresh cluster, mostly image pulls), then the API turns ready. Test and prod run but stay unready (no traffic) until a person promotes a model; the API notices within 30 s:

```bash
make registry-ui &    # registry on http://localhost:5001
CHURN_MLFLOW_TRACKING_URI=http://localhost:5001 make promote FROM=dev TO=staging
CHURN_MLFLOW_TRACKING_URI=http://localhost:5001 make promote FROM=staging TO=prod APPROVED_BY="Your Name" FAIRNESS_REVIEWED=1

KUBECONFIG=platform/.kube/telcoai.yaml kubectl -n telcoai-test port-forward svc/churn-api 18001:8000
curl localhost:18001/health     # {"status":"ok","model_version":"1"}
```

`make cluster-down` deletes everything.

## Layout

| Path                          | What                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `k3d/cluster.yaml`            | Cluster definition. It never touches `~/.kube/config` or the current context                      |
| `scripts/k3d.sh`              | Runs k3d locally, or inside the Rancher Desktop VM when the Windows Docker pipe is unavailable    |
| `scripts/promote-image.sh`    | Opens the image promotion PR                                                                      |
| `scripts/check-promotions.sh` | The promotion rules CI enforces                                                                   |
| `terraform/`                  | Namespaces (`ResourceQuota`, `LimitRange`), Argo CD, and the root application                     |
| `charts/mlflow-registry/`     | The shared MLflow registry                                                                        |
| `charts/churn-api/`           | Churn API serving one alias, plus the optional dev bootstrap job                                  |
| `gitops/apps/`                | What Argo CD deploys: the registry `Application` and the churn-api `ApplicationSet` (app of apps) |
| `gitops/envs/<env>/`          | Values per environment: the only files a deployment PR changes. A new folder is a new environment |

## Design decisions

- **Terraform installs the platform, Argo CD deploys the workloads.** Terraform owns things that change rarely and need a plan/apply review (namespaces, budgets, Argo CD itself). Services change daily and belong to GitOps, where every deployment is a commit and a rollback is `git revert`.
- **One registry, one alias per environment.** A model is never copied between environments; it is promoted. The version prod serves is the exact artifact that was tested, and its approval is recorded on it.
- **Immutable tags outside dev.** `main` moves; `sha-<commit>` never does. What runs in test and prod is readable from git, and prod can only run an image that was deployed to test first (CI reads the history of the test values).
- **Budgets per namespace.** Every namespace has a `ResourceQuota` and a `LimitRange`, so a container without resource settings still gets defaults, and one environment cannot starve another. The quota also counts the headroom a rolling update needs; the bootstrap job's CPU limit was lowered for this.
- **The bootstrap job only trains, and only once.** It is a sync hook (a plain Job's spec is immutable, so a new image would fail the sync) that exits at once if a `@dev` model exists. Training sets `@dev` only if the model passes the quality gate; the job never promotes.
- **Least privilege.** The API runs as a non-root user with a read-only root filesystem and no Linux capabilities; writable paths are `emptyDir` volumes.

## Found while testing on the cluster

- **MLflow was OOM-killed at 1.5 GiB.** MLflow 3.x starts a GenAI job runner next to the server: 8 more Python processes, 1.8 GiB in total instead of 0.6 GiB (measured with `docker stats`). The kernel log showed the kills; the container status only said `Error`, exit 137. A model registry does not need the runner, so `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`.
- **Default probes killed a healthy MLflow.** The 1 s default timeout is too short for a Python server starting under a CPU limit. A `startupProbe` covers the slow start, and the liveness probe only starts after it passes.
- **The API used to exit when its alias had no model**, so a fresh environment crash-looped, and after a promotion it could wait out the 5-minute restart back-off. Now it starts unready (`/health` 503, `/livez` 200 for the liveness probe) and follows its alias: on the cluster, moving an alias from version 1 to 2 switched the running pod in about 10 s, with no restart. The same path applies a rollback.
