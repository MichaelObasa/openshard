resource "google_secret_manager_secret" "db_password" {
  secret_id = "harbourdocs-db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret = google_secret_manager_secret.db_password.name
  # Deliberate flaw: placeholder value in source — in practice the secret_data
  # should be injected at apply time via a secrets pipeline, never committed as a value
  secret_data = "REPLACE_ME"
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "harbourdocs-api-key"

  replication {
    auto {}
  }

  # Missing: rotation block — no automatic rotation configured
}
