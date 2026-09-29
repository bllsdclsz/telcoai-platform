#!/bin/sh
# Opens the pull request that promotes a churn-api image to an environment.
#   promote-image.sh test [<commit>]   default: the latest commit on main
#   promote-image.sh prod              always the image test runs now
# Argo CD deploys it once the PR is merged; rollback is reverting that PR.
set -eu

env=${1:?usage: promote-image.sh test|prod [commit]}
values=platform/gitops/envs/$env/churn-api.yaml
[ "$env" = test ] || [ "$env" = prod ] || { echo "only test and prod are promoted by PR" >&2; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes first" >&2; exit 2; }

git fetch -q origin main
if [ "$env" = test ]; then
  sha=$(git rev-parse "${2:-origin/main}")
  tag=sha-$sha
else
  [ $# -lt 2 ] || { echo "prod takes the image test runs; promote the commit to test first" >&2; exit 2; }
  tag=$(git show "origin/main:platform/gitops/envs/test/churn-api.yaml" | sed -n 's/^  tag: *//p')
  sha=${tag#sha-}
fi
current=$(git show "origin/main:$values" | sed -n 's/^  tag: *//p')
[ "$tag" != "$current" ] || { echo "$env already runs $tag"; exit 0; }

short=$(echo "$sha" | cut -c1-7)
start=$(git rev-parse --abbrev-ref HEAD)
branch=promote-$env-$short
git switch -q -c "$branch" origin/main
sed -i "s/^  tag: .*/  tag: $tag/" "$values"
sh platform/scripts/check-promotions.sh || { git switch -q -f "$start"; git branch -q -D "$branch"; exit 1; }
git commit -q -m "Promote churn-api $short to $env" -- "$values"
git push -q -u origin "$branch"
gh pr create --base main --head "$branch" --title "Promote churn-api $short to $env" --body "$(cat <<EOF
Deploys \`telco-churn-api:$tag\` to **$env** (was \`${current:-none}\`).

Changes: $(git log --oneline "${current#sha-}..$sha" -- churn 2>/dev/null | wc -l | tr -d ' ') commits touching churn/ since the running image.
Argo CD syncs it after merge. Rollback: revert this PR.
EOF
)"
git switch -q "$start"
