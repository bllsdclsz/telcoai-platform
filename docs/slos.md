# Service level objectives

Every model-serving API on the platform (the churn API and every service generated from `templates/ml-service`) gets the same SLOs in every environment, from one rule set: `platform/charts/slo/files/rules/slo.yaml`.

## SLIs and objectives

| SLO          | SLI (what is measured)                              | Objective (30 days) | Error budget     |
| ------------ | --------------------------------------------------- | ------------------- | ---------------- |
| Availability | share of `/predict` requests that do not return 5xx | 99.5%               | 0.5% of requests |
| Latency      | share of `/predict` requests answered within 300 ms | 99%                 | 1% of requests   |

Why these:

- **Requests, not uptime.** A pod can be up and still fail every request (no model, broken model). Counting requests measures what callers see.
- **`/predict` only.** Health checks and metrics scrapes would inflate the SLI.
- **503 counts as a failure.** An API without a model answers 503; for its callers that is an outage, even though the pod is healthy.
- **300 ms** is the call-center requirement from Project 1 (a prediction while the agent is on the phone).

Dev, test and prod use the same objectives. Only prod pages a person (see routing below).

## Alerts: multi-window burn rates

The burn rate is how fast the budget is spent: 1 means exactly the whole budget in 30 days. Following the Google SRE workbook (chapter 5, "Alerting on SLOs"):

| Alert            | Condition                                          | Budget spent when it fires | Severity |
| ---------------- | -------------------------------------------------- | -------------------------- | -------- |
| `SLO*FastBurn`   | burn > 14.4 over 1 h **and** over 5 min, for 2 min | 2% in 1 h                  | critical |
| `SLO*SlowBurn`   | burn > 6 over 6 h **and** over 30 min, for 15 min  | 5% in 6 h                  | warning  |
| `ModelNotServed` | pod up for 10 min without a loaded model           | (not budget-based)         | warning  |

The long window keeps short spikes from paging; the short window makes the alert resolve soon after the problem stops. A fast burn inhibits the slow-burn alert for the same service, so one incident sends one alert.

The rules are unit-tested with `promtool test rules` (`platform/charts/slo/tests/`): a real fast burn fires with the right labels, 0.1% errors stays silent, a template service is named after its pod label, and a pod with a model never triggers `ModelNotServed`.

## Routing

Alertmanager (in `platform/monitoring/prometheus-values.yaml`):

- `env="prod"` **and** `severity="critical"` → `oncall-pager`
- everything else → `team-tickets`

Both receivers are placeholders; a real deployment adds PagerDuty and Slack/Jira configs there.

## Error budget policy

| Budget left (30 days) | What changes                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| > 25%                 | Normal: promote images and models as usual.                                                          |
| 0–25%                 | Promotions to prod need a note in the PR on why the change does not add risk.                        |
| exhausted             | Freeze: only fixes and rollbacks go to prod until the budget recovers. Reliability work comes first. |

The budget is a shared decision tool, not a punishment: if it is never spent, the objective is probably too loose or the team is shipping too slowly.

## Where to look

- **Dashboard:** `make grafana-ui` → http://localhost:3001, "TelcoAI SLOs" (per environment and service: availability, budget left, latency SLI, model served, burn rate).
- **Report:** `make prometheus-ui &` then `make slo-report` (or `WINDOW=7d`): a markdown table per environment and service, plus the alerts firing now.
- **Runbooks:** [runbooks/slo.md](runbooks/slo.md).
