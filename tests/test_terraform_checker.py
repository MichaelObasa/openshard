"""Tests for openshard.review.terraform_checker — deterministic Terraform static analysis."""

from __future__ import annotations

import textwrap
import unittest
from pathlib import Path

from openshard.review.terraform_checker import scan_terraform
from openshard.history.shard_contract import ShardFinding


def _write_tf(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestScanTerraformEmpty(unittest.TestCase):
    def test_no_tf_files_returns_empty(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = scan_terraform(Path(d))
        self.assertEqual(result, [])

    def test_clean_tf_returns_no_findings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", """
                resource "aws_s3_bucket" "good" {
                  bucket = "my-bucket"
                  tags = {
                    owner       = "platform"
                    environment = "prod"
                  }
                }
            """)
            result = scan_terraform(p)
        self.assertEqual(result, [])


class TestCIDRWildcard(unittest.TestCase):
    def _scan(self, content: str) -> list[ShardFinding]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "sg.tf", content)
            return scan_terraform(p)

    def test_ssh_port_22_wildcard_is_critical(self):
        findings = self._scan("""
            resource "aws_security_group_rule" "ssh" {
              ingress {
                from_port   = 22
                to_port     = 22
                cidr_blocks = ["0.0.0.0/0"]
              }
            }
        """)
        critical = [f for f in findings if f.severity == "Critical"]
        self.assertTrue(critical, "Expected Critical finding for SSH port 22 open to world")
        self.assertTrue(any("22" in f.message for f in critical))

    def test_rdp_port_3389_wildcard_is_critical(self):
        findings = self._scan("""
            resource "aws_security_group_rule" "rdp" {
              ingress {
                from_port   = 3389
                to_port     = 3389
                cidr_blocks = ["0.0.0.0/0"]
              }
            }
        """)
        critical = [f for f in findings if f.severity == "Critical"]
        self.assertTrue(critical, "Expected Critical finding for RDP port open to world")

    def test_no_finding_for_restricted_cidr(self):
        findings = self._scan("""
            resource "aws_security_group_rule" "ssh_internal" {
              ingress {
                from_port   = 22
                to_port     = 22
                cidr_blocks = ["10.0.0.0/8"]
              }
            }
        """)
        cidr_critical = [f for f in findings if f.severity == "Critical" and "CIDR" in f.message]
        self.assertEqual(cidr_critical, [])


class TestEncryption(unittest.TestCase):
    def test_encrypted_false_is_high(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "rds.tf", """
                resource "aws_db_instance" "db" {
                  encrypted = false
                  tags = {
                    owner       = "data"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        high = [f for f in findings if f.severity == "High" and "ncrypt" in f.message]
        self.assertTrue(high, "Expected High finding for encryption disabled")


class TestS3PublicACL(unittest.TestCase):
    def test_public_read_acl_is_critical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "s3.tf", """
                resource "aws_s3_bucket" "pub" {
                  bucket = "oops"
                  acl    = "public-read"
                  tags = {
                    owner       = "web"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        critical = [f for f in findings if f.severity == "Critical" and "S3" in f.message]
        self.assertTrue(critical, "Expected Critical finding for public S3 ACL")

    def test_private_acl_no_finding(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "s3.tf", """
                resource "aws_s3_bucket" "priv" {
                  bucket = "ok"
                  acl    = "private"
                  tags = {
                    owner       = "web"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        s3_critical = [f for f in findings if f.severity == "Critical" and "S3" in f.message]
        self.assertEqual(s3_critical, [])


class TestDeletionProtection(unittest.TestCase):
    def test_deletion_protection_false_is_high(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "db.tf", """
                resource "aws_db_instance" "db" {
                  deletion_protection = false
                  tags = {
                    owner       = "data"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        high = [f for f in findings if f.severity == "High" and "deletion_protection" in f.message]
        self.assertTrue(high, "Expected High finding for deletion_protection = false")


class TestIAMWildcards(unittest.TestCase):
    def test_wildcard_action_is_critical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "iam.tf", """
                resource "aws_iam_policy" "admin" {
                  policy = jsonencode({
                    Statement = [{
                      actions   = ["*"]
                      resources = ["arn:aws:s3:::my-bucket"]
                      Effect    = "Allow"
                    }]
                  })
                }
            """)
            findings = scan_terraform(p)
        critical = [f for f in findings if f.severity == "Critical" and "action" in f.message.lower()]
        self.assertTrue(critical, "Expected Critical finding for wildcard IAM action")

    def test_wildcard_resource_is_critical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "iam.tf", """
                resource "aws_iam_policy" "broad" {
                  policy = jsonencode({
                    Statement = [{
                      actions   = ["s3:GetObject"]
                      resources = ["*"]
                      Effect    = "Allow"
                    }]
                  })
                }
            """)
            findings = scan_terraform(p)
        critical = [f for f in findings if f.severity == "Critical" and "resource" in f.message.lower()]
        self.assertTrue(critical, "Expected Critical finding for wildcard IAM resource")


class TestMissingTags(unittest.TestCase):
    def test_no_tags_block_is_medium(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "ec2.tf", """
                resource "aws_instance" "web" {
                  ami           = "ami-123"
                  instance_type = "t3.micro"
                }
            """)
            findings = scan_terraform(p)
        medium = [f for f in findings if f.severity == "Medium"]
        self.assertTrue(medium, "Expected Medium finding for missing tags block")

    def test_missing_owner_tag_is_medium(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "ec2.tf", """
                resource "aws_instance" "web" {
                  ami           = "ami-123"
                  instance_type = "t3.micro"
                  tags = {
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        medium = [f for f in findings if f.severity == "Medium" and "owner" in f.message]
        self.assertTrue(medium, "Expected Medium finding for missing owner tag")

    def test_missing_environment_tag_is_medium(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "ec2.tf", """
                resource "aws_instance" "web" {
                  ami           = "ami-123"
                  instance_type = "t3.micro"
                  tags = {
                    owner = "team"
                  }
                }
            """)
            findings = scan_terraform(p)
        medium = [f for f in findings if f.severity == "Medium" and "environment" in f.message]
        self.assertTrue(medium, "Expected Medium finding for missing environment tag")


class TestBackendLocking(unittest.TestCase):
    def test_s3_backend_without_dynamo_is_high(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "backend.tf", """
                terraform {
                  backend "s3" {
                    bucket = "my-state-bucket"
                    key    = "prod/terraform.tfstate"
                    region = "us-east-1"
                  }
                }
            """)
            findings = scan_terraform(p)
        high = [f for f in findings if f.severity == "High" and "lock" in f.message.lower()]
        self.assertTrue(high, "Expected High finding for missing DynamoDB locking")

    def test_s3_backend_with_dynamo_no_finding(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "backend.tf", """
                terraform {
                  backend "s3" {
                    bucket         = "my-state-bucket"
                    key            = "prod/terraform.tfstate"
                    region         = "us-east-1"
                    dynamodb_table = "tf-state-lock"
                  }
                }
            """)
            findings = scan_terraform(p)
        lock_findings = [f for f in findings if "lock" in f.message.lower()]
        self.assertEqual(lock_findings, [])


class TestHardcodedSecrets(unittest.TestCase):
    def test_hardcoded_password_is_critical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "db.tf", """
                resource "aws_db_instance" "db" {
                  password = "super-secret-password-123"
                  tags = {
                    owner       = "data"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        critical = [f for f in findings if f.severity == "Critical" and "credential" in f.message.lower()]
        self.assertTrue(critical, "Expected Critical finding for hardcoded password")

    def test_variable_reference_not_flagged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "db.tf", """
                resource "aws_db_instance" "db" {
                  password = var.db_password
                  tags = {
                    owner       = "data"
                    environment = "prod"
                  }
                }
            """)
            findings = scan_terraform(p)
        secret_critical = [f for f in findings if "credential" in f.message.lower()]
        self.assertEqual(secret_critical, [])


class TestFlawedFixture(unittest.TestCase):
    """End-to-end: a deliberately flawed Terraform fixture produces >= 3 findings."""

    _FLAWED_MAIN_TF = """
        terraform {
          backend "s3" {
            bucket = "prod-state"
            key    = "prod/terraform.tfstate"
            region = "us-east-1"
          }
        }

        resource "aws_s3_bucket" "data" {
          bucket = "company-data"
          acl    = "public-read"
        }

        resource "aws_db_instance" "primary" {
          engine              = "postgres"
          instance_class      = "db.t3.micro"
          password            = "admin123!"
          encrypted           = false
          deletion_protection = false
        }

        resource "aws_security_group_rule" "ssh" {
          ingress {
            from_port   = 22
            to_port     = 22
            cidr_blocks = ["0.0.0.0/0"]
          }
        }

        resource "aws_iam_policy" "admin" {
          policy = jsonencode({
            Statement = [{
              actions   = ["*"]
              resources = ["*"]
              Effect    = "Allow"
            }]
          })
        }
    """

    def test_flawed_fixture_produces_at_least_3_findings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._FLAWED_MAIN_TF)
            findings = scan_terraform(p)
        self.assertGreaterEqual(
            len(findings), 3,
            f"Expected >= 3 findings, got {len(findings)}: {[f.message for f in findings]}"
        )

    def test_flawed_fixture_has_critical_findings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._FLAWED_MAIN_TF)
            findings = scan_terraform(p)
        critical = [f for f in findings if f.severity == "Critical"]
        self.assertTrue(critical, "Expected at least one Critical finding")

    def test_flawed_fixture_has_high_findings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._FLAWED_MAIN_TF)
            findings = scan_terraform(p)
        high = [f for f in findings if f.severity == "High"]
        self.assertTrue(high, "Expected at least one High finding")

    def test_findings_have_path_and_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._FLAWED_MAIN_TF)
            findings = scan_terraform(p)
        with_path = [f for f in findings if f.path]
        self.assertTrue(with_path, "Expected findings to carry file path")
        for f in with_path:
            self.assertIn("main.tf", f.path)

    def test_changed_files_stays_zero(self):
        """scan_terraform reads files; it must not write or modify anything."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._FLAWED_MAIN_TF)
            before = {fp: fp.read_bytes() for fp in p.rglob("*.tf")}
            scan_terraform(p)
            after = {fp: fp.read_bytes() for fp in p.rglob("*.tf")}
        self.assertEqual(before, after, "scan_terraform must not modify files")


class TestFindingsRenderInReceipt(unittest.TestCase):
    """Findings produced by the static checker render correctly in the receipt."""

    def test_findings_render_in_compact_receipt(self):
        from openshard.history.shard_contract import (
            ShardFinding,
            build_live_run_receipt,
            render_compact_shard_receipt,
        )
        findings = [
            ShardFinding(severity="Critical", message="Public S3 bucket detected", path="main.tf", line=5),
            ShardFinding(severity="High", message="Encryption disabled", path="main.tf", line=12),
        ]
        receipt = build_live_run_receipt(
            task="Review infra",
            run_id="test-123",
            run_index=1,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model=None,
            risk="High",
            sandbox="Not required",
            approval="Not required",
            estimated_cost=None,
            files_changed=0,
            verification_attempted=False,
            verification_passed=None,
            result_summary="2 issues found.",
            findings=findings,
        )
        rendered = render_compact_shard_receipt(receipt)
        self.assertIn("Changed", rendered)
        self.assertIn("0", rendered)
        self.assertIn("FINDINGS", rendered)

    def test_memo_says_found_x_issues(self):
        from openshard.cli.run_output import render_review_tldr_memo
        from openshard.history.shard_contract import ShardFinding
        findings = [
            ShardFinding(severity="Critical", message="Wildcard CIDR", path="sg.tf", line=3),
            ShardFinding(severity="High", message="Encryption off", path="db.tf", line=8),
            ShardFinding(severity="Medium", message="Missing tags", path="ec2.tf", line=1),
        ]
        memo = render_review_tldr_memo(findings, [])
        self.assertIn("Found 3 issues worth addressing.", memo)
        self.assertIn("Critical", memo)
        self.assertIn("High", memo)
        self.assertIn("Medium", memo)


class TestGCPSpecificChecks(unittest.TestCase):
    """GCP-specific deterministic checks added in the demo-readiness pass."""

    def _scan(self, content: str, name: str = "main.tf") -> list[ShardFinding]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, name, content)
            return scan_terraform(p)

    # ── IAM broad roles ──────────────────────────────────────────────────────

    def test_gcp_iam_editor_is_critical(self):
        findings = self._scan("""
            resource "google_project_iam_member" "api" {
              project = "proj"
              role    = "roles/editor"
              member  = "serviceAccount:x@p.iam.gserviceaccount.com"
            }
        """)
        critical = [f for f in findings if f.severity == "Critical" and "roles/editor" in f.message]
        self.assertTrue(critical, "Expected Critical for roles/editor")

    def test_gcp_iam_owner_is_critical(self):
        findings = self._scan("""
            resource "google_project_iam_binding" "admin" {
              project = "proj"
              role    = "roles/owner"
              members = ["user:admin@example.com"]
            }
        """)
        critical = [f for f in findings if f.severity == "Critical" and "roles/owner" in f.message]
        self.assertTrue(critical, "Expected Critical for roles/owner")

    def test_gcp_iam_camel_admin_is_high(self):
        findings = self._scan("""
            resource "google_project_iam_member" "sa" {
              project = "proj"
              role    = "roles/containerAdmin"
              member  = "serviceAccount:sa@proj.iam.gserviceaccount.com"
            }
        """)
        high = [f for f in findings if f.severity == "High" and "containerAdmin" in f.message]
        self.assertTrue(high, "Expected High for roles/containerAdmin")

    def test_gcp_iam_dotted_admin_is_high(self):
        findings = self._scan("""
            resource "google_project_iam_member" "sa" {
              project = "proj"
              role    = "roles/cloudsql.admin"
              member  = "serviceAccount:sa@proj.iam.gserviceaccount.com"
            }
        """)
        high = [f for f in findings if f.severity == "High" and "cloudsql.admin" in f.message]
        self.assertTrue(high, "Expected High for roles/cloudsql.admin")

    def test_gcp_iam_serviceaccount_admin_is_high(self):
        findings = self._scan("""
            resource "google_project_iam_member" "sa" {
              project = "proj"
              role    = "roles/iam.serviceAccountAdmin"
              member  = "serviceAccount:sa@proj.iam.gserviceaccount.com"
            }
        """)
        high = [f for f in findings if f.severity == "High" and "serviceAccountAdmin" in f.message]
        self.assertTrue(high, "Expected High for roles/iam.serviceAccountAdmin")

    def test_gcp_iam_viewer_not_flagged(self):
        findings = self._scan("""
            resource "google_project_iam_member" "viewer" {
              project = "proj"
              role    = "roles/viewer"
              member  = "serviceAccount:ro@proj.iam.gserviceaccount.com"
            }
        """)
        iam_findings = [f for f in findings if "roles/viewer" in f.message]
        self.assertEqual(iam_findings, [], "roles/viewer must not be flagged")

    # ── SQL authorized_networks ──────────────────────────────────────────────

    def test_gcp_sql_authorized_networks_wildcard_is_critical(self):
        findings = self._scan("""
            resource "google_sql_database_instance" "db" {
              name             = "db"
              database_version = "POSTGRES_14"
              region           = "us-central1"
              settings {
                tier = "db-g1-small"
                ip_configuration {
                  authorized_networks {
                    value = "0.0.0.0/0"
                  }
                }
              }
              deletion_protection = true
            }
        """)
        critical = [f for f in findings if f.severity == "Critical" and "authorized_networks" in f.message]
        self.assertTrue(critical, "Expected Critical for 0.0.0.0/0 in authorized_networks")

    def test_gcp_sql_deletion_protection_single_finding(self):
        """deletion_protection = false must produce exactly one finding, not two."""
        findings = self._scan("""
            resource "google_sql_database_instance" "db" {
              name             = "db"
              database_version = "POSTGRES_14"
              region           = "us-central1"
              settings { tier = "db-g1-small" }
              deletion_protection = false
            }
        """)
        dp_findings = [f for f in findings if "deletion_protection" in f.message.lower()]
        self.assertEqual(len(dp_findings), 1, "deletion_protection should produce exactly one finding")

    # ── Secret Manager ───────────────────────────────────────────────────────

    def test_gcp_secret_literal_is_critical(self):
        findings = self._scan("""
            resource "google_secret_manager_secret_version" "db_pass" {
              secret      = google_secret_manager_secret.db_pass.id
              secret_data = "SuperSecretPassword123!"
            }
        """)
        critical = [f for f in findings if f.severity == "Critical" and "secret_data" in f.message]
        self.assertTrue(critical, "Expected Critical for literal secret_data")

    def test_gcp_secret_var_ref_not_flagged(self):
        findings = self._scan("""
            resource "google_secret_manager_secret_version" "db_pass" {
              secret      = google_secret_manager_secret.db_pass.id
              secret_data = var.db_password
            }
        """)
        lit = [f for f in findings if "secret_data" in f.message]
        self.assertEqual(lit, [], "var. references must not be flagged as literal secrets")

    # ── Cloud Run ────────────────────────────────────────────────────────────

    def test_gcp_cloud_run_public_ingress_is_high(self):
        findings = self._scan("""
            resource "google_cloud_run_v2_service" "api" {
              name     = "api"
              location = "us-central1"
              ingress  = "INGRESS_TRAFFIC_ALL"
              template { containers { image = "gcr.io/p/api" } }
            }
        """)
        high = [f for f in findings if f.severity == "High" and "INGRESS_TRAFFIC_ALL" in f.message]
        self.assertTrue(high, "Expected High for INGRESS_TRAFFIC_ALL")

    # ── Storage ──────────────────────────────────────────────────────────────

    def test_gcp_storage_public_prevention_is_high(self):
        findings = self._scan("""
            resource "google_storage_bucket" "docs" {
              name                     = "docs"
              location                 = "US"
              public_access_prevention = "inherited"
            }
        """)
        high = [f for f in findings if f.severity == "High" and "public_access_prevention" in f.message]
        self.assertTrue(high, "Expected High for public_access_prevention = inherited")

    def test_gcp_storage_enforced_not_flagged(self):
        findings = self._scan("""
            resource "google_storage_bucket" "docs" {
              name                     = "docs"
              location                 = "US"
              public_access_prevention = "enforced"
              labels = { owner = "platform", environment = "prod" }
            }
        """)
        pub = [f for f in findings if "public_access_prevention" in f.message]
        self.assertEqual(pub, [], "enforced public_access_prevention must not be flagged")

    # ── Backend ──────────────────────────────────────────────────────────────

    def test_local_backend_is_high(self):
        findings = self._scan("""
            terraform {
              backend "local" {}
            }
        """)
        high = [f for f in findings if f.severity == "High" and "local backend" in f.message]
        self.assertTrue(high, "Expected High for local backend")

    # ── Missing labels (GCP allowlist) ───────────────────────────────────────

    def test_gcp_missing_labels_says_labels_not_tags(self):
        findings = self._scan("""
            resource "google_sql_database_instance" "db" {
              name             = "db"
              database_version = "POSTGRES_14"
              region           = "us-central1"
              settings { tier = "db-g1-small" }
              deletion_protection = true
            }
        """)
        label_findings = [f for f in findings if "medium" in f.severity.lower() or f.severity == "Medium"]
        label_findings = [f for f in label_findings if "sql_database_instance" in f.message]
        self.assertTrue(label_findings, "Expected a medium finding for missing labels")
        for f in label_findings:
            self.assertIn("labels", f.message.lower(), f"Message should say 'labels': {f.message}")
            self.assertNotIn("tags block", f.message.lower(), f"Message must not say 'tags block': {f.message}")

    def test_gcp_unsupported_label_resource_not_flagged(self):
        """google_project_iam_member is not on the labels allowlist — no labels finding."""
        findings = self._scan("""
            resource "google_project_iam_member" "viewer" {
              project = "proj"
              role    = "roles/viewer"
              member  = "serviceAccount:ro@proj.iam.gserviceaccount.com"
            }
        """)
        label_findings = [
            f for f in findings
            if "label" in f.message.lower() or "tags block" in f.message.lower()
        ]
        self.assertEqual(label_findings, [], "Non-allowlisted GCP resources must not produce label findings")

    def test_gcp_secret_version_not_flagged_for_labels(self):
        """google_secret_manager_secret_version is not on the labels allowlist."""
        findings = self._scan("""
            resource "google_secret_manager_secret_version" "v" {
              secret      = google_secret_manager_secret.x.id
              secret_data = var.secret_value
            }
        """)
        label_findings = [f for f in findings if "label" in f.message.lower() and "secret" in f.message.lower()]
        self.assertEqual(label_findings, [], "secret_version must not produce missing-label findings")


class TestGoldenDemoOutput(unittest.TestCase):
    """Recording-ready demo output for /pack production-iac-hardening on a GCP repo."""

    # DocuVault-style deliberately flawed GCP Terraform fixture
    _GCP_FLAWED_TF = """
        terraform {
          backend "local" {}
        }

        resource "google_project_iam_member" "api_admin" {
          project = "docuvault-prod"
          role    = "roles/editor"
          member  = "serviceAccount:api@docuvault-prod.iam.gserviceaccount.com"
        }

        resource "google_sql_database_instance" "docuvault_db" {
          name             = "docuvault-db"
          database_version = "POSTGRES_14"
          region           = "us-central1"
          settings {
            tier = "db-g1-small"
            ip_configuration {
              authorized_networks {
                value = "0.0.0.0/0"
              }
            }
            deletion_protection = false
          }
        }

        resource "google_secret_manager_secret_version" "db_password" {
          secret      = google_secret_manager_secret.db_password.id
          secret_data = "SuperSecretPassword123!"
        }

        resource "google_cloud_run_v2_service" "api" {
          name     = "docuvault-api"
          location = "us-central1"
          ingress  = "INGRESS_TRAFFIC_ALL"
          template { containers { image = "gcr.io/docuvault/api" } }
        }

        resource "google_storage_bucket" "documents" {
          name                     = "docuvault-documents"
          location                 = "US"
          public_access_prevention = "inherited"
        }

        resource "google_sql_database" "docuvault" {
          name     = "docuvault"
          instance = google_sql_database_instance.docuvault_db.name
        }

        resource "google_service_account" "api_sa" {
          account_id   = "api-service-account"
          display_name = "API Service Account"
        }

        resource "google_storage_bucket" "audit_logs" {
          name     = "docuvault-audit-logs"
          location = "US"
        }
    """

    def _findings(self) -> list[ShardFinding]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._GCP_FLAWED_TF)
            return scan_terraform(p)

    def _memo(self) -> str:
        from openshard.cli.run_output import render_review_tldr_memo
        return render_review_tldr_memo(self._findings(), [])

    def _render_last(self) -> str:
        from dataclasses import asdict
        import click
        from click.testing import CliRunner
        from openshard.cli.main import _render_log_entry
        findings = self._findings()
        entry = {
            "task": "/pack production-iac-hardening",
            "findings": [asdict(f) for f in findings],
            "is_review_task": True,
        }
        @click.command()
        def cmd():
            _render_log_entry(entry, "default")
        return CliRunner().invoke(cmd).output

    # ── Basic finding shape ──────────────────────────────────────────────────

    def test_fixture_has_critical_findings(self):
        critical = [f for f in self._findings() if f.severity == "Critical"]
        self.assertTrue(critical, "Expected at least one Critical finding")

    def test_fixture_has_high_findings(self):
        high = [f for f in self._findings() if f.severity == "High"]
        self.assertTrue(high, "Expected at least one High finding")

    # ── Memo content ────────────────────────────────────────────────────────

    def test_memo_header_uses_issue_count(self):
        memo = self._memo()
        self.assertIn("Found", memo)
        self.assertIn("worth addressing", memo)

    def test_memo_groups_metadata_findings(self):
        memo = self._memo()
        repeated_lines = [
            ln for ln in memo.splitlines()
            if "has no labels block" in ln or "has no tags block" in ln
        ]
        self.assertLessEqual(
            len(repeated_lines), 1,
            f"Expected at most 1 individual metadata line, got {len(repeated_lines)}: {repeated_lines}"
        )
        self.assertIn("resources are missing", memo, "Expected grouped metadata summary line")

    def test_memo_says_labels_not_tags_for_gcp(self):
        memo = self._memo()
        self.assertNotIn(
            "no tags block", memo,
            "GCP resources must not say 'tags block' — should say 'labels block'"
        )

    def test_memo_critical_before_medium(self):
        memo = self._memo()
        if "Critical" in memo and "Medium" in memo:
            self.assertLess(memo.index("Critical"), memo.index("Medium"))

    def test_memo_no_more_than_2_individual_medium_lines(self):
        memo = self._memo()
        medium_section_start = memo.find("\nMedium\n")
        if medium_section_start == -1:
            return
        after_medium = memo[medium_section_start + len("\nMedium\n"):]
        # Count lines before the next empty section or "+N more"
        medium_lines = []
        for ln in after_medium.splitlines():
            if not ln.strip() or ln.startswith("+") or (ln.strip() and ln[0].isalpha() and ln.strip() in ("Low", "Note")):
                break
            if ln.strip().startswith(("~", "-", "✓")):
                medium_lines.append(ln)
        self.assertLessEqual(
            len(medium_lines), 2,
            f"Expected at most 2 individual Medium lines, got: {medium_lines}"
        )

    def test_memo_no_structured_findings_leak(self):
        memo = self._memo()
        self.assertNotIn("STRUCTURED_FINDINGS", memo)
        self.assertNotIn('"summary":', memo)
        self.assertNotIn('"files":', memo)
        self.assertNotIn("run --workflow native", memo)
        self.assertNotIn("argv:", memo)
        self.assertNotIn("Files: 9 created", memo)

    def test_memo_line_budget(self):
        memo = self._memo()
        line_count = len(memo.splitlines())
        self.assertLessEqual(
            line_count, 22,
            f"Memo exceeds 22-line budget before receipt ({line_count} lines)"
        )

    def test_memo_raw_total_shown_when_capped(self):
        findings = self._findings()
        from openshard.history.shard_contract import group_review_findings
        _, _, hidden, raw_total = group_review_findings(findings)
        memo = self._memo()
        if hidden > 0:
            self.assertIn("raw findings recorded", memo)

    # ── Compact receipt ──────────────────────────────────────────────────────

    def test_compact_receipt_at_most_5_findings(self):
        from openshard.history.shard_contract import build_live_run_receipt, render_compact_shard_receipt
        findings = self._findings()
        receipt = build_live_run_receipt(
            task="/pack production-iac-hardening",
            run_id="golden-test-001",
            run_index=1,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model=None,
            risk="High",
            sandbox="Not required",
            files_changed=0,
            verification_attempted=False,
            verification_passed=None,
            approval="Not required",
            estimated_cost=0.0042,
            result_summary="demo",
            findings=findings,
        )
        rendered = render_compact_shard_receipt(receipt)
        finding_lines = [
            ln for ln in rendered.splitlines()
            if ln.strip().startswith(("✖", "⚠", "~", "X ", "! ", "+ "))
        ]
        self.assertLessEqual(len(finding_lines), 5, f"Compact receipt has > 5 finding lines: {finding_lines}")

    def test_compact_receipt_has_more_line_when_over_5(self):
        from openshard.history.shard_contract import (
            ShardFinding, build_live_run_receipt, render_compact_shard_receipt,
        )
        many = [ShardFinding(severity="High", message=f"Issue {i}") for i in range(10)]
        receipt = build_live_run_receipt(
            task="test",
            run_id="x",
            run_index=1,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model=None,
            risk="High",
            sandbox="Not required",
            files_changed=0,
            verification_attempted=False,
            verification_passed=None,
            approval="Not required",
            estimated_cost=None,
            result_summary="10 issues",
            findings=many,
        )
        rendered = render_compact_shard_receipt(receipt)
        self.assertIn("more findings recorded", rendered)

    def test_compact_receipt_changed_zero(self):
        from openshard.history.shard_contract import build_live_run_receipt, render_compact_shard_receipt
        receipt = build_live_run_receipt(
            task="/pack production-iac-hardening",
            run_id="golden-zero",
            run_index=1,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model=None,
            risk="High",
            sandbox="Not required",
            files_changed=0,
            verification_attempted=False,
            verification_passed=None,
            approval="Not required",
            estimated_cost=0.001,
            result_summary="review done",
        )
        rendered = render_compact_shard_receipt(receipt)
        self.assertIn("0 file", rendered)

    # ── /last default rendering ──────────────────────────────────────────────

    def test_review_complete_appears_exactly_once(self):
        out = self._render_last()
        count = out.count("Review complete")
        self.assertEqual(count, 1, f"'Review complete' should appear exactly once, got {count}")

    def test_last_contains_found_issues(self):
        out = self._render_last()
        self.assertIn("Found", out)
        self.assertIn("worth addressing", out)

    def test_last_no_structured_findings_leak(self):
        out = self._render_last()
        self.assertNotIn("STRUCTURED_FINDINGS", out)
        self.assertNotIn('"summary":', out)
        self.assertNotIn("run --workflow native", out)
        self.assertNotIn("Files: 9 created", out)
        self.assertNotIn("no structured findings captured", out)

    # ── Immutability ─────────────────────────────────────────────────────────

    def test_changed_stays_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", self._GCP_FLAWED_TF)
            before = {fp: fp.read_bytes() for fp in p.rglob("*.tf")}
            scan_terraform(p)
            after = {fp: fp.read_bytes() for fp in p.rglob("*.tf")}
        self.assertEqual(before, after, "scan_terraform must not modify files")

    # ── Phase 2 polish acceptance tests ─────────────────────────────────────

    def test_no_duplicate_medium_header(self):
        """When metadata noise AND substantive Medium findings coexist, only one Medium header."""
        from openshard.history.shard_contract import ShardFinding
        from openshard.cli.run_output import render_review_tldr_memo
        findings = [
            ShardFinding(severity="Critical", message="IAM role too broad — use least privilege"),
            ShardFinding(severity="Medium", message="TLS 1.0 enabled — disable in favor of TLS 1.2+"),
            ShardFinding(severity="Medium", message="Resource google_storage_bucket.x has no labels block — add owner and environment labels"),
            ShardFinding(severity="Medium", message="Resource google_compute_instance.y has no labels block — add owner and environment labels"),
        ]
        memo = render_review_tldr_memo(findings, [])
        count = memo.count("\nMedium\n")
        self.assertLessEqual(count, 1, f"'Medium' header appeared {count} times — expected exactly 1:\n{memo}")

    def test_receipt_result_uses_two_count_format(self):
        """build_shard_receipt() with review findings must produce two-count Result."""
        from dataclasses import asdict
        from openshard.history.shard_contract import build_shard_receipt
        findings = self._findings()
        entry = {
            "task": "/pack production-iac-hardening",
            "findings": [asdict(f) for f in findings],
            "is_review_task": True,
            "form_factor": {"read_only": True},
        }
        receipt = build_shard_receipt(entry, 1)
        self.assertIn("issue areas found", receipt.result, f"Expected two-count format, got: {receipt.result!r}")
        self.assertIn("raw findings recorded", receipt.result, f"Expected raw count, got: {receipt.result!r}")

    def test_compact_receipt_ellipsis_long_finding(self):
        """Finding messages longer than 79 chars are word-safely truncated with ellipsis."""
        from openshard.history.shard_contract import build_live_run_receipt, render_compact_shard_receipt, ShardFinding
        long_msg = "A" * 40 + " " + "B" * 40  # 81 chars with a word boundary at 40
        receipt = build_live_run_receipt(
            task="test",
            run_id="x",
            run_index=1,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model=None,
            risk="High",
            sandbox="Off",
            files_changed=0,
            verification_attempted=False,
            verification_passed=None,
            approval="Not required",
            estimated_cost=None,
            result_summary="1 issue",
            findings=[ShardFinding(severity="High", message=long_msg)],
        )
        rendered = render_compact_shard_receipt(receipt)
        finding_line = next(
            (ln for ln in rendered.splitlines() if "⚠" in ln or "!" in ln),
            None,
        )
        self.assertIsNotNone(finding_line, "Expected a finding line in receipt")
        self.assertIn("…", finding_line, f"Expected ellipsis for long message, got: {finding_line!r}")
        self.assertNotIn("B" * 40, finding_line, "Should not contain the full second word (was cut)")

    def test_review_complete_not_in_render_post_run_output(self):
        """render_post_run() must NOT echo 'Review complete' — pipeline.py owns that."""
        import click
        from click.testing import CliRunner
        from openshard.cli.run_output import render_post_run
        findings = self._findings()

        @click.command()
        def cmd():
            render_post_run(
                stage_runs=[],
                routing_decision=None,
                verification_attempted=False,
                verification_passed=None,
                readonly_task=True,
                validator_policy=None,
                validator_result=None,
                final_files=[],
                detail="default",
                notes=[],
                repo_facts=None,
                task="/pack production-iac-hardening",
                run_id="test-001",
                run_index=1,
                risk="High",
                sandbox="Off",
                approval="Not required",
                is_native=True,
                exec_result_summary="",
                findings=findings,
                is_review_task=True,
            )

        out = CliRunner().invoke(cmd).output
        self.assertNotIn("Review complete", out, "render_post_run() must not echo 'Review complete'; pipeline.py does that")

    def test_sandbox_off_in_strip(self):
        """TUI status strip must say Sandbox [OFF]."""
        from openshard.tui.app import _STATUS_STRIP
        self.assertIn("Sandbox [OFF]", _STATUS_STRIP, f"Expected 'Sandbox [OFF]' in strip: {_STATUS_STRIP!r}")

    def test_sandbox_off_in_live_receipt(self):
        """render_post_run() with readonly_task=True → compact receipt shows Sandbox Off."""
        import click
        from click.testing import CliRunner
        from openshard.cli.run_output import render_post_run

        @click.command()
        def cmd():
            render_post_run(
                stage_runs=[],
                routing_decision=None,
                verification_attempted=False,
                verification_passed=None,
                readonly_task=True,
                validator_policy=None,
                validator_result=None,
                final_files=[],
                detail="default",
                notes=[],
                repo_facts=None,
                task="review",
                run_id="test-002",
                run_index=1,
                risk="High",
                sandbox="Off",
                approval="Not required",
                is_native=True,
                exec_result_summary="review done",
                is_review_task=True,
            )

        out = CliRunner().invoke(cmd).output
        sandbox_line = next((ln for ln in out.splitlines() if "Sandbox" in ln), None)
        self.assertIsNotNone(sandbox_line, "Expected a Sandbox row in receipt")
        self.assertIn("Off", sandbox_line, f"Sandbox row must say 'Off': {sandbox_line!r}")
        self.assertNotIn("Not required", sandbox_line, f"Sandbox row must not say 'Not required': {sandbox_line!r}")

    def test_sandbox_off_in_last_receipt(self):
        """build_shard_receipt() with read_only form_factor → receipt.sandbox == 'Off'."""
        from openshard.history.shard_contract import build_shard_receipt
        entry = {
            "task": "review",
            "is_review_task": True,
            "form_factor": {"read_only": True},
        }
        receipt = build_shard_receipt(entry, 1)
        self.assertEqual(receipt.sandbox, "Off", f"Expected sandbox='Off', got {receipt.sandbox!r}")

    def test_sandbox_off_in_last_receipt_via_is_review_task(self):
        """build_shard_receipt() with is_review_task=True (no read_only flag) → sandbox 'Off'.

        Pack runs may not classify as read_only by text heuristic but have is_review_task logged.
        """
        from openshard.history.shard_contract import build_shard_receipt
        entry = {
            "task": "/pack production-iac-hardening",
            "is_review_task": True,
        }
        receipt = build_shard_receipt(entry, 1)
        self.assertEqual(receipt.sandbox, "Off", f"Expected sandbox='Off' for review task, got {receipt.sandbox!r}")

    def test_live_receipt_result_two_count_not_truncated(self):
        """render_post_run() with findings → Result shows full two-count, not first sentence only.

        _result_display() would truncate "9 issue areas found. 24 raw findings recorded."
        at the first '. ' — we bypass it via the result= parameter.
        """
        import click
        from click.testing import CliRunner
        from openshard.cli.run_output import render_post_run
        findings = self._findings()

        @click.command()
        def cmd():
            render_post_run(
                stage_runs=[],
                routing_decision=None,
                verification_attempted=False,
                verification_passed=None,
                readonly_task=False,
                validator_policy=None,
                validator_result=None,
                final_files=[],
                detail="default",
                notes=[],
                repo_facts=None,
                task="/pack production-iac-hardening",
                run_id="test-result-001",
                run_index=1,
                risk="High",
                sandbox="Off",
                approval="Not required",
                is_native=True,
                exec_result_summary="",
                findings=findings,
                is_review_task=True,
            )

        out = CliRunner().invoke(cmd).output
        result_line = next((ln for ln in out.splitlines() if "Result" in ln), None)
        self.assertIsNotNone(result_line, "Expected a Result row in receipt")
        self.assertIn("raw findings recorded", result_line,
                      f"Result must contain 'raw findings recorded': {result_line!r}")
        self.assertIn("issue areas found", result_line,
                      f"Result must contain 'issue areas found': {result_line!r}")

    def test_last_receipt_result_two_count(self):
        """build_shard_receipt() with review findings → Result contains both counts."""
        from dataclasses import asdict
        from openshard.history.shard_contract import build_shard_receipt
        findings = self._findings()
        entry = {
            "task": "/pack production-iac-hardening",
            "findings": [asdict(f) for f in findings],
            "is_review_task": True,
        }
        receipt = build_shard_receipt(entry, 1)
        self.assertIn("issue areas found", receipt.result,
                      f"Expected two-count in /last result: {receipt.result!r}")
        self.assertIn("raw findings recorded", receipt.result,
                      f"Expected raw count in /last result: {receipt.result!r}")

    def test_model_line_hidden_in_more_mode(self):
        """_render_log_entry() with detail='more' must not print a standalone 'Model:' line."""
        import click
        from click.testing import CliRunner
        from openshard.cli.main import _render_log_entry
        from dataclasses import asdict
        findings = self._findings()
        entry = {
            "task": "/pack production-iac-hardening",
            "findings": [asdict(f) for f in findings],
            "is_review_task": True,
            "stage_runs": [
                {"model": "claude-sonnet-4-5-20251001", "stage_type": "implementation", "duration": 12.3, "cost": 0.0042},
            ],
        }

        @click.command()
        def cmd():
            _render_log_entry(entry, "more")

        out = CliRunner().invoke(cmd).output
        standalone_model_lines = [
            ln for ln in out.splitlines()
            if ln.startswith("Model:") or ln.startswith("Models:")
        ]
        self.assertEqual(
            standalone_model_lines, [],
            f"Found standalone Model: line(s) in --more output: {standalone_model_lines}",
        )


class TestProviderDetection(unittest.TestCase):
    """detect_terraform_providers() identifies cloud providers from .tf content."""

    def _detect(self, content: str) -> frozenset[str]:
        from openshard.review.terraform_checker import detect_terraform_providers
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", content)
            return detect_terraform_providers(p)

    def test_detects_gcp_from_provider_block(self):
        providers = self._detect('provider "google" { project = "my-proj" }')
        self.assertIn("gcp", providers)

    def test_detects_gcp_from_google_beta_provider(self):
        providers = self._detect('provider "google-beta" { project = "my-proj" }')
        self.assertIn("gcp", providers)

    def test_detects_gcp_from_resource_prefix(self):
        providers = self._detect('resource "google_storage_bucket" "assets" { name = "x" }')
        self.assertIn("gcp", providers)

    def test_detects_aws_from_provider_block(self):
        providers = self._detect('provider "aws" { region = "us-east-1" }')
        self.assertIn("aws", providers)

    def test_detects_aws_from_resource_prefix(self):
        providers = self._detect('resource "aws_s3_bucket" "data" { bucket = "x" }')
        self.assertIn("aws", providers)

    def test_detects_both_when_mixed(self):
        providers = self._detect("""
            resource "google_compute_instance" "vm" { name = "x" }
            resource "aws_s3_bucket" "backup" { bucket = "y" }
        """)
        self.assertIn("gcp", providers)
        self.assertIn("aws", providers)

    def test_empty_for_unknown_provider(self):
        providers = self._detect('resource "null_resource" "noop" {}')
        self.assertEqual(providers, frozenset())

    def test_no_tf_files_returns_empty(self):
        from openshard.review.terraform_checker import detect_terraform_providers
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = detect_terraform_providers(Path(d))
        self.assertEqual(result, frozenset())


class TestProviderGating(unittest.TestCase):
    """Provider-specific checks are gated; unknown provider emits only generic findings."""

    def _scan(self, content: str) -> list[ShardFinding]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_tf(p, "main.tf", content)
            return scan_terraform(p)

    def test_gcp_only_repo_no_s3_acl_finding(self):
        # A GCP repo with a local `acl` variable must not produce an S3 ACL finding.
        findings = self._scan("""
            resource "google_storage_bucket" "assets" {
              name     = "my-assets"
              location = "US"
              labels   = { owner = "platform", environment = "prod" }
            }
            locals {
              acl = "public-read"
            }
        """)
        s3_acl = [f for f in findings if "S3" in f.message or "public acl" in f.message.lower() or "public-" in f.message.lower()]
        self.assertEqual(s3_acl, [], f"Unexpected S3 ACL finding in GCP-only repo: {s3_acl}")

    def test_gcp_only_repo_no_dynamodb_finding(self):
        # A GCP repo with local backend must get the local-backend finding but NOT the DynamoDB advice.
        findings = self._scan("""
            terraform {
              backend "local" {}
            }
            resource "google_compute_instance" "vm" {
              name         = "vm"
              machine_type = "e2-micro"
              zone         = "us-central1-a"
              labels       = { owner = "ops", environment = "prod" }
              boot_disk { initialize_params { image = "debian-cloud/debian-11" } }
              network_interface { network = "default" }
            }
        """)
        dynamo = [f for f in findings if "dynamodb" in f.message.lower()]
        self.assertEqual(dynamo, [], f"Unexpected DynamoDB finding in GCP-only repo: {dynamo}")
        local_backend = [f for f in findings if "local backend" in f.message.lower()]
        self.assertTrue(local_backend, "Expected local-backend finding to still fire for GCP repo")

    def test_aws_only_repo_no_gcp_findings(self):
        # An AWS-only repo must not produce GCP-specific findings.
        findings = self._scan("""
            resource "aws_s3_bucket" "data" {
              bucket = "my-bucket"
              tags   = { owner = "platform", environment = "prod" }
            }
        """)
        gcp_findings = [f for f in findings if "google_" in f.message.lower() or "gcp" in f.message.lower()]
        self.assertEqual(gcp_findings, [], f"Unexpected GCP finding in AWS-only repo: {gcp_findings}")

    def test_unknown_provider_no_s3_finding(self):
        # When provider is unknown, S3 ACL check must not run.
        findings = self._scan("""
            resource "null_resource" "noop" {}
            locals {
              acl = "public-read"
            }
        """)
        s3_acl = [f for f in findings if "S3" in f.message or "public-" in f.message.lower()]
        self.assertEqual(s3_acl, [], f"Unexpected S3 finding for unknown provider: {s3_acl}")

    def test_unknown_provider_no_dynamodb_finding(self):
        # Unknown provider with local backend: only the generic local-backend finding fires.
        findings = self._scan("""
            terraform {
              backend "local" {}
            }
            resource "null_resource" "noop" {}
        """)
        dynamo = [f for f in findings if "dynamodb" in f.message.lower()]
        self.assertEqual(dynamo, [], f"Unexpected DynamoDB finding for unknown provider: {dynamo}")

    def test_aws_provider_s3_acl_finding_fires(self):
        # With AWS detected, S3 public ACL check must still fire.
        findings = self._scan("""
            resource "aws_s3_bucket" "bad" {
              bucket = "leaky"
              acl    = "public-read"
              tags   = { owner = "ops", environment = "prod" }
            }
        """)
        s3 = [f for f in findings if f.severity == "Critical" and "public" in f.message.lower()]
        self.assertTrue(s3, "Expected S3 public ACL finding for AWS repo")

    def test_aws_provider_dynamodb_locking_fires(self):
        # With AWS detected, missing DynamoDB locking on S3 backend must still fire.
        findings = self._scan("""
            provider "aws" { region = "us-east-1" }
            terraform {
              backend "s3" {
                bucket = "tf-state"
                key    = "prod/terraform.tfstate"
                region = "us-east-1"
              }
            }
            resource "aws_s3_bucket" "state" {
              bucket = "tf-state"
              tags   = { owner = "ops", environment = "prod" }
            }
        """)
        dynamo = [f for f in findings if "dynamodb" in f.message.lower()]
        self.assertTrue(dynamo, "Expected DynamoDB locking finding for AWS repo with S3 backend")


if __name__ == "__main__":
    unittest.main()
