variable "kubeconfig" {
  description = "Kubeconfig of the platform cluster (written by `make cluster-up`, never the default one)."
  type        = string
  default     = "../.kube/telcoai.yaml"
}

variable "namespaces" {
  description = "Namespaces with their resource budgets: one per environment, plus the shared model registry."
  type = map(object({
    cpu_requests    = string
    memory_requests = string
    cpu_limits      = string
    memory_limits   = string
    pods            = number
  }))
  default = {
    dev  = { cpu_requests = "2", memory_requests = "4Gi", cpu_limits = "4", memory_limits = "6Gi", pods = 20 }
    test = { cpu_requests = "2", memory_requests = "4Gi", cpu_limits = "4", memory_limits = "6Gi", pods = 20 }
    prod = { cpu_requests = "3", memory_requests = "6Gi", cpu_limits = "6", memory_limits = "8Gi", pods = 30 }
    # One MLflow registry for all environments: each environment serves its own alias
    # (dev -> @dev, test -> @staging, prod -> @prod), so a model is promoted, never copied.
    registry = { cpu_requests = "500m", memory_requests = "1Gi", cpu_limits = "2", memory_limits = "2Gi", pods = 5 }
    # Prometheus, Alertmanager and Grafana for every environment's SLOs.
    monitoring = { cpu_requests = "500m", memory_requests = "768Mi", cpu_limits = "3", memory_limits = "2Gi", pods = 10 }
  }
}

variable "argocd_chart_version" {
  description = "argo-cd Helm chart version (app v3.5.3)."
  type        = string
  default     = "10.9.4"
}

variable "argocd_apps_chart_version" {
  description = "argocd-apps Helm chart version, used for the root application."
  type        = string
  default     = "2.0.6"
}

variable "gitops_repo_url" {
  description = "Git repository Argo CD watches."
  type        = string
  default     = "https://github.com/bllsdclsz/telcoai-platform.git"
}

variable "gitops_revision" {
  description = "Branch Argo CD deploys from: a merge to main is a deployment."
  type        = string
  default     = "main"
}
