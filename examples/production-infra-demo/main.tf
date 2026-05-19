terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    # Deliberate flaw: hard-coded bucket name — should use a variable or workspace prefix
    bucket = "docuvault-tf-state-000000"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  # Deliberate flaw: region hard-coded here instead of using var.region
  region  = "europe-west2"
}

locals {
  service_name = "docuvault"
  environment  = var.environment
}
