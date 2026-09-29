terraform {
  required_version = ">= 1.16.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "3.2.1"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "3.3.0"
    }
  }

  # Local state: this module manages a disposable local cluster. A shared environment would use
  # a remote backend with locking (e.g. an azurerm or s3 backend).
}

provider "kubernetes" {
  config_path = var.kubeconfig
}

provider "helm" {
  kubernetes = {
    config_path = var.kubeconfig
  }
}
