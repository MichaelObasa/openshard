output "api_url" {
  description = "DocuVault API Cloud Run service URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "processor_url" {
  description = "DocuVault Processor Cloud Run service URL"
  value       = google_cloud_run_v2_service.processor.uri
}

# Deliberate flaw: Cloud SQL connection name is missing
# Developers need this to configure Cloud SQL Auth Proxy or the Cloud Run DB connection string
# output "db_connection_name" {
#   description = "Cloud SQL instance connection name for Auth Proxy"
#   value       = google_sql_database_instance.docuvault_db.connection_name
# }
