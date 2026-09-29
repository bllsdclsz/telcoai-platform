# Runbook: SLO alerts (all model-serving APIs)

Every alert carries `env`, `service` and `namespace` labels. Objectives and the budget policy: [../slos.md](../slos.md).

First look, for any of them:

```bash
make grafana-ui        # "TelcoAI SLOs": pick env and service
make argocd-ui         # did a sync just happen?
KUBECONFIG=platform/.kube/telcoai.yaml kubectl -n telcoai-<env> get pods
KUBECONFIG=platform/.kube/telcoai.yaml kubectl -n telcoai-<env> logs deploy/<service> --since=30m
```

## SLOAvailabilityFastBurn

**Meaning:** more than 7.2% of `/predict` requests failed over the last hour (and still in the last 5 min). At this rate the 30-day budget is gone in about two days.

1. **What changed?** Most incidents follow a change. Check, in this order:
   - a model promotion: the _Model served_ panel shows the version; `churn_model_info` changed recently?
   - an image promotion: the last merged `Promote …` PR for this env (`git log -- platform/gitops/envs/<env>/`).
2. **Model change → roll back the alias** (takes effect within ~30 s, no deploy):
   `CHURN_MLFLOW_TRACKING_URI=http://localhost:5001 make promote FROM=<alias>-previous TO=<alias> APPROVED_BY="<you>"` (after `make registry-ui`).
3. **Image change → revert the promotion PR** and merge it; Argo CD syncs the previous image.
4. **No change?** 503s with `waiting_for_model`: see _ModelNotServed_. 5xx with a stack trace: an input the model can't handle; capture the request from the logs and open an issue.
5. Once the burn stops, the alert resolves within minutes (short window). Write a postmortem if the budget took a visible hit.

## SLOAvailabilitySlowBurn

**Meaning:** more than 3% of requests failed over 6 hours. Not urgent, but the budget will not last the month.

Same diagnosis as above, during working hours. Often a subset of inputs that fails (a category the model has never seen), or intermittent registry or node trouble.

## SLOLatencyFastBurn

**Meaning:** more than 14.4% of requests took longer than 300 ms over the last hour.

1. Traffic spike? _Requests by status_ panel. Batch callers should send at most 1,000 records per request.
2. New model version? A bigger model is slower: roll the alias back as above and compare.
3. CPU throttling: `kubectl top pod -n telcoai-<env>`; the API's CPU limit is 1 core. Scale replicas in the env values file (a PR) if the traffic is real.

## SLOLatencySlowBurn

**Meaning:** more than 6% of requests slower than 300 ms over 6 hours. Usually gradual: growing batch sizes or a heavier model. Profile one request; raise it with the model owner.

## ModelNotServed

**Meaning:** the API pod has been up for 10 minutes without a model. Readiness keeps traffic away, so callers get no answer from this environment.

1. Which alias does the env serve? dev → `@dev`, test → `@staging`, prod → `@prod`.
2. `make registry-ui` and check the aliases of the model. Alias missing → someone has to promote a model (prod: a named approver, see `platform/README.md`).
3. Alias set but not loading → pod logs show `model check for @… failed`: registry unreachable (check `telcoai-registry`), or the model can't be loaded by this image (a dependency missing from the image).
