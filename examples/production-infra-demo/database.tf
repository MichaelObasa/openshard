resource "google_sql_database_instance" "docuvault_db" {
  name             = "docuvault-db"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      # Deliberate flaw: public IP enabled with no authorized networks restriction
      ipv4_enabled = true
      # Missing: private_network = google_compute_network.docuvault_vpc.id
      # Missing: authorized_networks { value = "10.0.0.0/8" }
    }

    backup_configuration {
      enabled = true
    }
  }

  # Deliberate flaw: deletion protection disabled — a DROP DATABASE at 2am is unrecoverable
  deletion_protection = false
}

resource "google_sql_database" "docuvault" {
  name     = "docuvault"
  instance = google_sql_database_instance.docuvault_db.name
}

resource "google_sql_user" "app_user" {
  name     = "docuvault-app"
  instance = google_sql_database_instance.docuvault_db.name
  password = var.db_password
}
