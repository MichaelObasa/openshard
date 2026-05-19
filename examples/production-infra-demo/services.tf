resource "google_cloud_run_v2_service" "api" {
  name     = "docuvault-api"
  location = var.region

  template {
    service_account = google_service_account.api_sa.email

    containers {
      image = "europe-west2-docker.pkg.dev/docuvault-dev-000000/docuvault/api:latest"

      env {
        name  = "DB_HOST"
        value = google_sql_database_instance.docuvault_db.public_ip_address
      }

      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }
    }

    # Deliberate flaw: no VPC connector — Cloud Run egresses via public internet
    # Missing: vpc_access { connector = ... egress = "ALL_TRAFFIC" }

    scaling {
      # Deliberate flaw: no min_instance_count — cold starts will hit a 2am incident
      max_instance_count = 10
    }
  }
}

resource "google_cloud_run_v2_service" "processor" {
  name     = "docuvault-processor"
  location = var.region

  template {
    service_account = google_service_account.processor_sa.email

    containers {
      image = "europe-west2-docker.pkg.dev/docuvault-dev-000000/docuvault/processor:latest"

      env {
        name  = "DB_HOST"
        value = google_sql_database_instance.docuvault_db.public_ip_address
      }
    }

    scaling {
      max_instance_count = 5
    }
  }
}
