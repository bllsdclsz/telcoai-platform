# Platform: Kubernetes, Terraform, GitOps

The hub layer: product teams ship a Helm chart and an image; the platform gives them an environment with a budget and deploys whatever `main` says. Nobody runs `kubectl apply` by hand.

```
GitHub (main) ──── churn-ci: test → image → ghcr.io/…/telco-churn-api:{main, sha-…}
      │
      ▼ polled by
Argo CD ── root app "platform-apps" (platform/gitops/apps/) ── churn-dev ──► namespace telcoai-dev
   ▲                                                                           ├─ MLflow registry (PVC)
   │ installed by                                                              ├─ churn API (serves @prod)
Terraform ── namespaces telcoai-{dev,test,prod} with quotas + container defaults └─ bootstrap job (dev only)
   ▲
k3d cluster "telcoai" (1 server + 1 agent)
```

## Run it

Needs k3d, Terraform, Helm and kubectl (`scoop install k3d terraform helm kubectl`) and a Docker engine (Rancher Desktop works).

```bash
make cluster-up       # k3d cluster + platform/.kube/telcoai.yaml
make platform-apply   # namespaces, quotas, Argo CD, root app
make argocd-ui        # prints the admin password, then http://localhost:8081
```

Argo CD then syncs `churn-dev` on its own. The first sync takes a minute or two: the dev bootstrap job trains a model and promotes it, then the API turns ready.

```bash
KUBECONFIG=platform/.kube/telcoai.yaml kubectl -n telcoai-dev port-forward svc/churn-api 18000:8000
curl localhost:18000/health     # {"status":"ok","model_version":"1"}
```

`make cluster-down` deletes everything.

## Layout

| Path                  | What                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `k3d/cluster.yaml`    | Cluster definition. It never touches `~/.kube/config` or the current context                   |
| `scripts/k3d.sh`      | Runs k3d locally, or inside the Rancher Desktop VM when the Windows Docker pipe is unavailable |
| `terraform/`          | Environments (namespace, `ResourceQuota`, `LimitRange`), Argo CD, and the root application     |
| `charts/churn-stack/` | Helm chart: MLflow registry, churn API, optional bootstrap job                                 |
| `gitops/apps/`        | Argo CD `Application`s, one per service and environment (app of apps)                          |
| `gitops/envs/<env>/`  | Values per environment: the only files a deployment PR changes                                 |

## Design decisions

- **Terraform installs the platform, Argo CD deploys the workloads.** Terraform owns things that change rarely and need a plan/apply review (namespaces, budgets, Argo CD itself). Services change daily and belong to GitOps, where every deployment is a commit and a rollback is `git revert`.
- **Budgets per environment.** Every namespace has a `ResourceQuota` and a `LimitRange`, so a container without resource settings still gets defaults, and one environment cannot starve another. The quota also counts the headroom a rolling update needs; the bootstrap job's CPU limit was lowered for this.
- **The API only serves `@prod`.** Dev gets there through a bootstrap job that trains (the quality gate still applies) and promotes with a named approver. The chart refuses to render the job without one. Test and prod have no bootstrap: their models arrive through the human-approved `churn promote`.
- **The bootstrap job is a sync hook, and idempotent.** A plain Job's spec is immutable, so a new image would fail the sync. As a hook it is recreated on each sync, and it exits at once if the serving alias already exists, so a sync never retrains.
- **Least privilege.** The API runs as a non-root user with a read-only root filesystem and no Linux capabilities; writable paths are `emptyDir` volumes.

## Found while testing on the cluster

- **MLflow was OOM-killed at 1.5 GiB.** MLflow 3.x starts a GenAI job runner with seven more Python processes (about 1.4 GiB) next to the server. The kernel log showed the kills. The container status only said `Error`, exit 137. A model registry does not need the runner: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` brings it under 1 GiB.
- **Default probes killed a healthy MLflow.** The 1 s default timeout is too short for a Python server starting under a CPU limit. A `startupProbe` covers the slow start, and the liveness probe only starts after it passes.
