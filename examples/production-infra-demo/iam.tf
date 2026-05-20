resource "google_service_account" "api_sa" {
  account_id   = "harbourdocs-api-sa"
  display_name = "HarbourDocs API Service Account"
}

resource "google_service_account" "processor_sa" {
  account_id   = "harbourdocs-processor-sa"
  display_name = "HarbourDocs Processor Service Account"
}

# Deliberate flaw: roles/editor is far too broad for a service account
# Should use minimal roles: roles/cloudsql.client, roles/storage.objectAdmin, etc.
resource "google_project_iam_member" "api_editor" {
  project = var.project_id
  role    = "roles/editor"
  member  = "serviceAccount:${google_service_account.api_sa.email}"
  # Missing: condition block to scope to specific resources
}

# Deliberate flaw: processor also gets roles/editor — no least-privilege separation
resource "google_project_iam_member" "processor_editor" {
  project = var.project_id
  role    = "roles/editor"
  member  = "serviceAccount:${google_service_account.processor_sa.email}"
}
