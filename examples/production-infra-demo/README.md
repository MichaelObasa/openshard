# HarbourDocs — Production-Infra Demo

**HarbourDocs** is a fictional document-processing platform used as a sanitised demo scenario for OpenShard.

The infrastructure provisions a GCP workload with:
- Two Cloud Run services (`harbourdocs-api`, `harbourdocs-processor`)
- Cloud SQL PostgreSQL (`harbourdocs-db`)
- Service accounts and IAM bindings
- Secret Manager secrets
- Networking (VPC, subnets, Cloud NAT, firewall rules)
- Storage (GCS buckets)
- Monitoring (alert policies, log sink)
- GCS remote state backend

## Purpose

This codebase is **deliberately flawed**. It is designed to be used as the input for the OpenShard flagship demo task: an IaC hardening review across three lenses — security, operability, and developer experience.

See `demo-task.md` for the task text to paste into the OpenShard TUI.

## Identifiers

All names, project IDs, resource IDs, CIDRs, and accounts are fake and public-safe:

| Field | Value |
|---|---|
| Project ID | `harbourdocs-dev-000000` |
| Region | `europe-west2` |
| State bucket | `harbourdocs-tf-state-000000` |
| Services | `harbourdocs-api`, `harbourdocs-processor` |
| Database | `harbourdocs-db` |
| IP ranges | RFC 1918 private ranges (`10.x.x.x`) |
| Secrets | Placeholder values only (`REPLACE_ME`) |

## Public demo note

Do not replace these fake identifiers with real project IDs, real IPs, real secrets, or real company names before sharing or recording. The point of this repo is that it is safe to show publicly.

## Files

| File | Contents |
|---|---|
| `demo-task.md` | The flagship hardening review task — paste into OpenShard TUI |
| `main.tf` | Backend, provider, locals |
| `variables.tf` | Input variables |
| `network.tf` | VPC, subnets, Cloud NAT, firewall rules |
| `database.tf` | Cloud SQL PostgreSQL instance |
| `iam.tf` | Service accounts and IAM bindings |
| `services.tf` | Cloud Run services |
| `storage.tf` | GCS buckets |
| `secrets.tf` | Secret Manager secrets |
| `monitoring.tf` | Log sink and alert policy |
| `outputs.tf` | Service URLs and connection info |
| `terraform.tfvars.example` | Template for local variable values |
