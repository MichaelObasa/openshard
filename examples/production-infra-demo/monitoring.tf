resource "google_logging_project_sink" "audit_sink" {
  name        = "docuvault-audit-sink"
  destination = "storage.googleapis.com/${google_storage_bucket.exports.name}"
  filter      = "resource.type = \"cloud_run_revision\""
}

resource "google_monitoring_alert_policy" "error_rate" {
  display_name = "DocuVault API Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Error count above threshold"

    condition_threshold {
      filter     = "resource.type = \"cloud_run_revision\" AND metric.type = \"run.googleapis.com/request_count\""
      duration   = "60s"
      comparison = "COMPARISON_GT"

      # Deliberate flaw: threshold at 0 — fires on any error, guaranteed alert fatigue
      threshold_value = 0

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  # Deliberate flaw: no notification channels — alerts fire but no one is paged
  notification_channels = []

  # Missing: uptime check resource for the API health endpoint
}
