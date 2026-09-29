#!/bin/sh
# Promotion rules for the image each service runs (CI runs this on every platform PR):
#   - test and prod pin an immutable tag: sha-<40 hex>, never a moving one like "main"
#   - the tag exists in the registry
#   - prod only runs an image that has been deployed to test (now or in git history)
# Services are the files in platform/gitops/envs/{test,prod}/<service>.yaml; the image
# repository comes from platform/charts/<service>/values.yaml.
set -eu

envs_dir=platform/gitops/envs
fail=0

tag_of() { sed -n 's/^  tag: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "$1"; }
repo_of() { sed -n 's#^  repository: *ghcr.io/\([^ ]*\) *$#\1#p' "platform/charts/$1/values.yaml"; }

image_exists() { # <owner/name> <tag>
  token=$(curl -sf "https://ghcr.io/token?scope=repository:$1:pull" | sed 's/.*"token":"\([^"]*\)".*/\1/')
  curl -sfo /dev/null -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/$1/manifests/$2"
}

for values in "$envs_dir"/test/*.yaml "$envs_dir"/prod/*.yaml; do
  [ -f "$values" ] || continue
  env=$(basename "$(dirname "$values")")
  service=$(basename "$values" .yaml)
  tag=$(tag_of "$values")
  if ! echo "$tag" | grep -Eq '^sha-[0-9a-f]{40}$'; then
    echo "::error::$service/$env must pin an immutable image tag (sha-<commit>), found '$tag'"
    fail=1; continue
  fi
  repo=$(repo_of "$service")
  if image_exists "$repo" "$tag"; then echo "$service/$env: $tag exists"; else
    echo "::error::$service/$env: image $repo:$tag not found in ghcr.io"; fail=1
  fi
  [ "$env" = prod ] || continue
  test_values="$envs_dir/test/$service.yaml"
  if { [ -f "$test_values" ] && tag_of "$test_values"; git log -p --format= -- "$test_values" \
    | sed -n 's/^+  tag: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p'; } | grep -qx "$tag"; then
    echo "$service/prod: $tag has been in test"
  else
    echo "::error::$service/prod runs $tag, which was never deployed to test. Promote it to test first."
    fail=1
  fi
done

exit $fail
