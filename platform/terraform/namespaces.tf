# One namespace per environment, plus the shared model registry. The quota caps what an environment can consume, so a runaway
# deployment in dev can't starve prod; the limit range gives every container defaults, so no
# pod runs unbounded.

resource "kubernetes_namespace_v1" "env" {
  for_each = var.namespaces

  metadata {
    name = "telcoai-${each.key}"
    labels = {
      "app.kubernetes.io/part-of" = "telcoai"
      "telcoai.example/namespace" = each.key
    }
  }
}

resource "kubernetes_resource_quota_v1" "env" {
  for_each = var.namespaces

  metadata {
    name      = "budget"
    namespace = kubernetes_namespace_v1.env[each.key].metadata[0].name
  }
  spec {
    hard = {
      "requests.cpu"    = each.value.cpu_requests
      "requests.memory" = each.value.memory_requests
      "limits.cpu"      = each.value.cpu_limits
      "limits.memory"   = each.value.memory_limits
      pods              = each.value.pods
    }
  }
}

resource "kubernetes_limit_range_v1" "env" {
  for_each = var.namespaces

  metadata {
    name      = "container-defaults"
    namespace = kubernetes_namespace_v1.env[each.key].metadata[0].name
  }
  spec {
    limit {
      type = "Container"
      default_request = {
        cpu    = "100m"
        memory = "128Mi"
      }
      default = {
        cpu    = "500m"
        memory = "512Mi"
      }
    }
  }
}
