output "namespaces" {
  description = "Environment namespaces."
  value       = [for ns in kubernetes_namespace_v1.env : ns.metadata[0].name]
}

output "argocd_ui" {
  description = "How to open the Argo CD UI."
  value       = "make argocd-ui  (then http://localhost:8081, user admin)"
}

output "argocd_admin_password" {
  description = "Initial admin password (local cluster only)."
  value       = "kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
}
