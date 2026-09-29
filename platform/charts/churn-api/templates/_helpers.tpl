{{- define "churn.name" -}}{{ .Release.Name }}{{- end -}}

{{- define "churn.labels" -}}
app.kubernetes.io/part-of: telcoai
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "churn.image" -}}
{{- $tag := required "image.tag is required (dev: main; test and prod: sha-<commit>)" .Values.image.tag -}}
{{ .Values.image.repository }}:{{ $tag }}
{{- end -}}
