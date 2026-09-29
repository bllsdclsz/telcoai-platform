{{- define "registry.labels" -}}
app.kubernetes.io/part-of: telcoai
app.kubernetes.io/name: mlflow
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "registry.hosts" -}}
{{- $svc := .Release.Name -}}
{{- $ns := .Release.Namespace -}}
{{- $hosts := list $svc (printf "%s:5000" $svc) (printf "%s.%s:5000" $svc $ns) (printf "%s.%s.svc.cluster.local:5000" $svc $ns) "localhost:5000" -}}
{{- concat $hosts .Values.allowedHosts | join "," -}}
{{- end -}}
