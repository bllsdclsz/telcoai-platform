# Argo CD, and one root application that deploys everything declared in platform/gitops/apps.
# After this, deployments happen through Git: merge to main, Argo CD syncs.

resource "helm_release" "argocd" {
  name             = "argocd"
  namespace        = "argocd"
  create_namespace = true
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-cd"
  version          = var.argocd_chart_version
  wait             = true
  timeout          = 600

  values = [yamlencode({
    configs = {
      params = { "server.insecure" = true } # local cluster: reached via port-forward, no TLS
      cm     = { "timeout.reconciliation" = "60s" }
    }
    dex            = { enabled = false }
    notifications  = { enabled = false }
    applicationSet = { replicas = 1 }
    server         = { resources = { requests = { cpu = "50m", memory = "128Mi" } } }
    repoServer     = { resources = { requests = { cpu = "50m", memory = "128Mi" } } }
    controller     = { resources = { requests = { cpu = "100m", memory = "256Mi" } } }
  })]
}

resource "helm_release" "root_app" {
  name       = "platform-root"
  namespace  = "argocd"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argocd-apps"
  version    = var.argocd_apps_chart_version
  depends_on = [helm_release.argocd] # the Application CRD comes with Argo CD

  values = [yamlencode({
    applications = {
      platform-apps = {
        namespace = "argocd"
        project   = "default"
        source = {
          repoURL        = var.gitops_repo_url
          targetRevision = var.gitops_revision
          path           = "platform/gitops/apps"
        }
        destination = {
          server    = "https://kubernetes.default.svc"
          namespace = "argocd"
        }
        syncPolicy = {
          automated   = { prune = true, selfHeal = true }
          syncOptions = ["CreateNamespace=false"]
        }
      }
    }
  })]
}
