#!/bin/sh
# Lints every chart and validates what it renders for every environment that deploys it
# (platform/gitops/envs/<env>/<chart>.yaml), plus the Argo CD apps. Needs helm and kubeconform.
set -eu

k8s_version=1.35.0
crds='https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'

for chart in platform/charts/*/; do
  name=$(basename "$chart")
  found=0
  for values in platform/gitops/envs/*/"$name".yaml; do
    [ -f "$values" ] || continue
    found=1
    env=$(basename "$(dirname "$values")")
    echo "== $name ($env)"
    helm lint --quiet --strict "$chart" -f "$values"
    helm template "$name" "$chart" -n "telcoai-$env" -f "$values" \
      | kubeconform -strict -summary -kubernetes-version "$k8s_version"
  done
  if [ "$found" = 0 ]; then # platform charts deployed once, with their defaults
    echo "== $name"
    helm lint --quiet --strict "$chart"
    helm template "$name" "$chart" | kubeconform -strict -summary -kubernetes-version "$k8s_version"
  fi
done

echo "== Argo CD apps"
kubeconform -strict -summary -schema-location default -schema-location "$crds" platform/gitops/apps
