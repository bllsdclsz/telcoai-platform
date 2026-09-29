#!/bin/sh
# Opens the pull request that promotes a service's image to an environment.
#   promote-image.sh <service> test [<commit>]   default: the latest commit on main
#   promote-image.sh <service> prod              always the image test runs now
# Argo CD deploys it once the PR is merged; rollback is reverting that PR. A service's first
# promotion to an environment creates its values file there (and so its Argo CD app).
set -eu

service=${1:?usage: promote-image.sh <service> test|prod [commit]}
env=${2:?usage: promote-image.sh <service> test|prod [commit]}
values=platform/gitops/envs/$env/$service.yaml
[ "$env" = test ] || [ "$env" = prod ] || { echo "only test and prod are promoted by PR" >&2; exit 2; }
[ -d "platform/charts/$service" ] || { echo "no chart platform/charts/$service" >&2; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes first" >&2; exit 2; }

git fetch -q origin main
if [ "$env" = test ]; then
  sha=$(git rev-parse "${3:-origin/main}")
  tag=sha-$sha
else
  [ $# -lt 3 ] || { echo "prod takes the image test runs; promote the commit to test first" >&2; exit 2; }
  tag=$(git show "origin/main:platform/gitops/envs/test/$service.yaml" | sed -n 's/^  tag: *//p')
  sha=${tag#sha-}
fi
current=$(git show "origin/main:$values" 2>/dev/null | sed -n 's/^  tag: *//p' || true)
[ "$tag" != "$current" ] || { echo "$service already runs $tag in $env"; exit 0; }

short=$(echo "$sha" | cut -c1-7)
start=$(git rev-parse --abbrev-ref HEAD)
branch=promote-$service-$env-$short
git switch -q -c "$branch" origin/main
if [ -f "$values" ]; then
  sed -i "s/^  tag: .*/  tag: $tag/" "$values"
else
  alias=$([ "$env" = test ] && echo staging || echo prod)
  printf '# Pinned image, changed only by promotion PRs (make promote-image). Serves @%s.\nimage:\n  tag: %s\n\napi:\n  servingAlias: %s\n' \
    "$alias" "$tag" "$alias" > "$values"
  git add "$values"
fi
sh platform/scripts/check-promotions.sh || { git switch -q -f "$start"; git branch -q -D "$branch"; exit 1; }
git commit -q -m "Promote $service $short to $env" -- "$values"
git push -q -u origin "$branch"
src=$([ -d "$service" ] && echo "$service" || echo churn)
changes=$( [ -n "$current" ] && git log --oneline "${current#sha-}..$sha" -- "$src" 2>/dev/null | wc -l | tr -d ' ' || echo "?")
gh pr create --base main --head "$branch" --title "Promote $service $short to $env" --body "$(cat <<EOF
Deploys \`$service:$tag\` to **$env** (was \`${current:-not deployed}\`).

Commits touching \`$src/\` since the running image: $changes.
Argo CD syncs it after merge. Rollback: revert this PR.
EOF
)"
git switch -q "$start"
