# ml-service template

A new model-serving service, already wired into the platform, in one command:

```bash
make new-service NAME=fraud-score     # asks for description, team, model name
```

It writes:

| Generated                                                  | What it gives the team                                                                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `<name>/src/<package>/app.py`                              | FastAPI service that serves `models:/<model>@<alias>` from the shared registry and follows the alias (no restart on promotion or rollback) |
| `<name>/tests/`                                            | Tests for predictions, readiness before a model exists, alias moves and registry outages                                                   |
| `<name>/Dockerfile`                                        | Non-root image, dependencies cached separately from code                                                                                   |
| `<name>/docs/model-card.md`, `<name>/eval/thresholds.yaml` | The model card an approver reads before prod, and the quality gate to enforce in training                                                  |
| `.github/workflows/<name>-ci.yml`                          | ruff, mypy, pytest, image build; publishes `ghcr.io/<owner>/<name>:{main, sha-<commit>}` on main                                           |
| `platform/charts/<name>/`                                  | Helm chart: `/livez` and `/health` probes, resource limits, read-only root filesystem, Prometheus annotations                              |
| `platform/gitops/apps/<name>.yaml`                         | Argo CD ApplicationSet: one app per environment that has a values file                                                                     |
| `platform/gitops/envs/dev/<name>.yaml`                     | Dev deploys from `main` right after the first merge                                                                                        |

Then the platform's normal flow applies: register a model as `@dev` and dev turns ready; `make promote-image SERVICE=<name> ENV=test` (then `ENV=prod`) creates the test and prod deployments by pull request, and CI enforces the promotion rules for every service.

## Decisions

- **The platform conventions are in the code a team starts from**, not in a wiki: liveness vs readiness, alias following, fail-fast registry calls, metrics names, security context. A team changes what it predicts, not how it runs.
- **Each service is its own uv project** with its own lock file. The workspace pins some packages for Feast; a new service shouldn't inherit that.
- **Copier, with answers recorded:** `<name>/.copier-answers.yml` keeps the inputs a service was generated with. Moving the template to its own repository would also enable `copier update`, which brings later template changes into existing services as a reviewable diff.
- **The template is tested like code:** `template-ci` generates a service on every template change and runs its lint, type check, tests, chart validation and image build.
