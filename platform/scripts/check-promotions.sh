#!/bin/sh
# Promotion rules for the image each environment runs (CI runs this on every platform PR):
#   - test and prod pin an immutable tag: sha-<40 hex>, never a moving one like "main"
#   - the tag exists in the registry
#   - prod only runs an image that has been deployed to test (now or in git history)
set -eu

envs_dir=platform/gitops/envs
repo=bllsdclsz/telco-churn-api
fail=0

tag_of() { sed -n 's/^  tag: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "$1"; }

image_exists() {
  token=$(curl -sf "https://ghcr.io/token?scope=repository:$repo:pull" | sed 's/.*"token":"\([^"]*\)".*/\1/')
  curl -sfo /dev/null -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/$repo/manifests/$1"
}

for env in test prod; do
  tag=$(tag_of "$envs_dir/$env/churn-api.yaml")
  if ! echo "$tag" | grep -Eq '^sha-[0-9a-f]{40}$'; then
    echo "::error::$env must pin an immutable image tag (sha-<commit>), found '$tag'"; fail=1; continue
  fi
  if image_exists "$tag"; then echo "$env: $tag exists"; else
    echo "::error::$env: image $repo:$tag not found in ghcr.io"; fail=1
  fi
done

prod_tag=$(tag_of "$envs_dir/prod/churn-api.yaml")
if { tag_of "$envs_dir/test/churn-api.yaml"; git log -p --format= -- "$envs_dir/test/churn-api.yaml" \
  | sed -n 's/^+  tag: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p'; } | grep -qx "$prod_tag"; then
  echo "prod: $prod_tag has been in test"
else
  echo "::error::prod runs $prod_tag, which was never deployed to test. Promote it to test first."; fail=1
fi

exit $fail
