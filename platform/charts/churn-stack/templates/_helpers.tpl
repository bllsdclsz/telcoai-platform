{{- define "churn.name" -}}{{ .Release.Name }}{{- end -}}

{{- define "churn.labels" -}}
app.kubernetes.io/part-of: telcoai
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "churn.mlflowUrl" -}}http://{{ include "churn.name" . }}-mlflow:5000{{- end -}}
