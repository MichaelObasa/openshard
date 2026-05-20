resource "google_storage_bucket" "documents" {
  name     = "harbourdocs-documents-000000"
  location = var.region

  # Deliberate flaw: ACLs still active — object-level permissions bypass bucket policy
  uniform_bucket_level_access = false

  # Deliberate flaw: no lifecycle rules — no cost control or data retention enforcement
}

resource "google_storage_bucket" "exports" {
  name     = "harbourdocs-exports-000000"
  location = var.region

  uniform_bucket_level_access = false
}
