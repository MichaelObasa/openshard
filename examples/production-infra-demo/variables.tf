variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region for resource deployment"
  default     = "europe-west2"
}

variable "environment" {
  type        = string
  description = "Deployment environment (dev, staging, prod)"
  default     = "dev"
}

variable "db_password" {
  type        = string
  description = "Cloud SQL database password"
  sensitive   = true
}
