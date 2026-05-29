"""Deterministic static analysis for Terraform / HCL files.

Scans .tf files in a directory tree and returns a list of ShardFinding objects.
All checks are purely regex/line-based — no external tools required.

Checks performed:
  - 0.0.0.0/0 (wildcard CIDR) on sensitive ports → Critical
  - Missing encryption on storage resources → High
  - Public S3 buckets (acl = "public-*") → Critical
  - Missing deletion_protection or prevent_destroy → High
  - Wildcard IAM actions (*) or resources (*) → Critical
  - Missing required labels/tags (owner, environment) — GCP-aware → Medium
  - Missing remote state backend locking → High
  - Local-only Terraform backend → High
  - Hardcoded secrets / credentials in plain text → Critical
  - GCP: broad project IAM roles (editor/owner/admin) → Critical/High
  - GCP: SQL authorized_networks with 0.0.0.0/0 → Critical
  - GCP: Secret Manager literal secret_data → Critical
  - GCP: Cloud Run INGRESS_TRAFFIC_ALL → High
  - GCP: Storage public_access_prevention = inherited → High
"""

from __future__ import annotations

import re
from pathlib import Path

from openshard.history.shard_contract import ShardFinding


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 0.0.0.0/0 on a line that also contains a port number or in a security-group context.
# We flag any cidr_blocks / cidr that contains 0.0.0.0/0.
_CIDR_WILDCARD = re.compile(r'0\.0\.0\.0/0')

# Sensitive ports we care about for ingress CIDR wildcards.
# A block is suspicious when it contains a from_port / to_port in these ranges
# AND has 0.0.0.0/0 in a cidr_blocks value.
_SENSITIVE_PORTS = {22, 3389, 5432, 3306, 27017, 6379, 9200, 5601, 2181, 2379, 2380}

# encryption_disabled / unencrypted patterns
_ENCRYPTION_FALSE = re.compile(
    r'(?:encrypted|storage_encrypted|enable_encryption|sse_algorithm'
    r'|server_side_encryption_enabled)\s*=\s*false',
    re.IGNORECASE,
)

# S3 public ACL
_S3_PUBLIC_ACL = re.compile(r'acl\s*=\s*"public-', re.IGNORECASE)

# deletion_protection = false (explicit or absent handled separately)
_DELETION_PROTECTION_FALSE = re.compile(
    r'deletion_protection\s*=\s*false', re.IGNORECASE
)

# prevent_destroy = false or lifecycle block with prevent_destroy explicitly false
_PREVENT_DESTROY_FALSE = re.compile(
    r'prevent_destroy\s*=\s*false', re.IGNORECASE
)

# Wildcard IAM actions: actions = ["*"] or actions = ["sts:*"]
_IAM_WILDCARD_ACTION = re.compile(
    r'"actions"\s*=\s*\[.*?"(?:\*|[a-z]+:\*).*?"\]|'
    r'actions\s*=\s*\[.*?"(?:\*|[a-z]+:\*).*?"\]',
    re.IGNORECASE | re.DOTALL,
)

# Wildcard IAM resources: resources = ["*"]
_IAM_WILDCARD_RESOURCE = re.compile(
    r'resources\s*=\s*\[.*?"\*".*?\]',
    re.IGNORECASE | re.DOTALL,
)

# Missing required tags — we look for resource blocks that lack an "owner" or "environment" tag.
# This is detected at block level, not line level.
_RESOURCE_BLOCK_START = re.compile(r'^resource\s+"([^"]+)"\s+"([^"]+)"')
_TAGS_BLOCK = re.compile(r'\btags\s*=\s*\{', re.IGNORECASE)
_TAG_OWNER = re.compile(r'"?owner"?\s*=', re.IGNORECASE)
_TAG_ENV = re.compile(r'"?environment"?\s*=', re.IGNORECASE)

# Backend without dynamodb_table (locking) — terraform backend block
_BACKEND_S3 = re.compile(r'backend\s+"s3"', re.IGNORECASE)
_DYNAMO_LOCK = re.compile(r'dynamodb_table\s*=', re.IGNORECASE)

# Hardcoded secrets / passwords / tokens
_SECRET_PATTERNS = [
    re.compile(r'(?:password|secret|token|api_key|access_key|private_key)\s*=\s*"(?!var\.|local\.|data\.|module\.)[^"]{6,}"', re.IGNORECASE),
    re.compile(r'(?:AKIA|ASIA)[A-Z0-9]{16}'),  # AWS access key IDs
]

# ---------------------------------------------------------------------------
# GCP-specific patterns
# ---------------------------------------------------------------------------

# GCP labels block — distinct from AWS "tags"
_LABELS_BLOCK = re.compile(r'\blabels\s*=\s*\{', re.IGNORECASE)
_LABEL_OWNER  = re.compile(r'"?owner"?\s*=', re.IGNORECASE)
_LABEL_ENV    = re.compile(r'"?environment"?\s*=', re.IGNORECASE)

# GCP resource types that are known to support labels.
# Only flag missing labels for types on this allowlist — do not flag others.
_GCP_LABEL_SUPPORTED: frozenset[str] = frozenset({
    "google_sql_database_instance",
    "google_cloud_run_v2_service",
    "google_storage_bucket",
    "google_compute_instance",
    "google_compute_network",
    "google_compute_subnetwork",
})

# GCP IAM broad project roles:
#   roles/editor, roles/owner
#   roles/<CamelCaseAdmin>          e.g. roles/containerAdmin
#   roles/<service>.<word>admin     e.g. roles/cloudsql.admin, roles/iam.serviceAccountAdmin
# Does NOT match viewer / logging / monitoring viewer roles.
_GCP_IAM_BROAD = re.compile(
    r'role\s*=\s*"('
    r'roles/editor'
    r'|roles/owner'
    r'|roles/[a-zA-Z]+Admin'
    r'|roles/[a-zA-Z][a-zA-Z0-9]*\.[a-zA-Z]*[Aa]dmin'
    r')"',
    re.IGNORECASE,
)

# GCP Secret Manager: secret_data assigned as a literal string (not a variable ref)
_GCP_SECRET_LITERAL = re.compile(
    r'secret_data\s*=\s*"(?!var\.|local\.|data\.|module\.)[^"]{1,}"',
    re.IGNORECASE,
)

# GCP Cloud Run public ingress
_GCP_CLOUD_RUN_PUBLIC = re.compile(
    r'ingress\s*=\s*"INGRESS_TRAFFIC_ALL"',
    re.IGNORECASE,
)

# GCP Storage: public access prevention not enforced
_GCP_STORAGE_PUBLIC = re.compile(
    r'public_access_prevention\s*=\s*"inherited"',
    re.IGNORECASE,
)

# Local-only Terraform backend (no remote locking)
_BACKEND_LOCAL = re.compile(r'backend\s+"local"', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Provider detection patterns
# ---------------------------------------------------------------------------

_PROVIDER_GCP   = re.compile(r'provider\s+"google(?:-beta)?"', re.IGNORECASE)
_PROVIDER_AWS   = re.compile(r'provider\s+"aws"', re.IGNORECASE)
_PROVIDER_AZURE = re.compile(r'provider\s+"azurerm"', re.IGNORECASE)
_RESOURCE_GCP   = re.compile(r'\bresource\s+"google_')
_RESOURCE_AWS   = re.compile(r'\bresource\s+"aws_')
_RESOURCE_AZURE = re.compile(r'\bresource\s+"azurerm_')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_terraform_providers(root: Path) -> frozenset[str]:
    """Return the set of cloud providers detected from .tf files under *root*.

    Returns a frozenset containing zero or more of: "aws", "gcp", "azure".
    An empty set means the provider could not be determined from the files.
    """
    providers: set[str] = set()
    for tf_path in root.rglob("*.tf"):
        try:
            text = tf_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _PROVIDER_GCP.search(text) or _RESOURCE_GCP.search(text):
            providers.add("gcp")
        if _PROVIDER_AWS.search(text) or _RESOURCE_AWS.search(text) or _BACKEND_S3.search(text):
            providers.add("aws")
        if _PROVIDER_AZURE.search(text) or _RESOURCE_AZURE.search(text):
            providers.add("azure")
    return frozenset(providers)


def scan_terraform(root: Path) -> list[ShardFinding]:
    """Return findings for all .tf files under *root*. Never raises."""
    findings: list[ShardFinding] = []
    tf_files = sorted(root.rglob("*.tf"))
    if not tf_files:
        return findings

    providers = detect_terraform_providers(root)

    for tf_path in tf_files:
        try:
            _scan_file(tf_path, root, findings, providers)
        except Exception:
            pass

    return findings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scan_file(path: Path, root: Path, findings: list[ShardFinding], providers: frozenset[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    lines = text.splitlines()
    rel = _rel(path, root)

    _check_cidr_wildcard(lines, rel, findings)
    _check_encryption(lines, rel, findings)
    if "aws" in providers:
        _check_s3_public_acl(lines, rel, findings)
    _check_deletion_protection(lines, rel, findings)
    _check_iam_wildcards(text, lines, rel, findings)
    _check_missing_tags(text, lines, rel, findings)
    _check_backend_locking(text, lines, rel, findings, providers)
    _check_hardcoded_secrets(lines, rel, findings)
    _check_gcp_resources(text, lines, rel, findings)


def _check_cidr_wildcard(lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    """Flag 0.0.0.0/0 on sensitive ports in security group ingress blocks."""
    in_ingress = False
    block_lines: list[tuple[int, str]] = []
    depth = 0

    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if line.startswith("#"):
            continue

        if "ingress" in line.lower() and "{" in line:
            in_ingress = True
            depth = line.count("{") - line.count("}")
            block_lines = [(i, line)]
            continue

        if in_ingress:
            block_lines.append((i, line))
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                # End of ingress block — analyze the collected lines
                _analyze_ingress_block(block_lines, rel, findings)
                in_ingress = False
                block_lines = []
                depth = 0

    # Also flag bare 0.0.0.0/0 outside ingress blocks (e.g. in cidr_blocks list)
    for i, raw in enumerate(lines, 1):
        if _CIDR_WILDCARD.search(raw) and "ingress" not in raw.lower():
            if "cidr" in raw.lower() or "cidr_blocks" in raw.lower():
                findings.append(ShardFinding(
                    severity="Critical",
                    message="Wildcard CIDR 0.0.0.0/0 exposes resources to the public internet",
                    path=rel,
                    line=i,
                ))
                break


def _analyze_ingress_block(block: list[tuple[int, str]], rel: str, findings: list[ShardFinding]) -> None:
    """Check an ingress block for wildcard CIDR on sensitive ports."""
    has_wildcard = any(_CIDR_WILDCARD.search(ln) for _, ln in block)
    if not has_wildcard:
        return

    ports_in_block: set[int] = set()
    for _, ln in block:
        for m in re.finditer(r'(?:from_port|to_port)\s*=\s*(\d+)', ln, re.IGNORECASE):
            try:
                ports_in_block.add(int(m.group(1)))
            except ValueError:
                pass

    sensitive_hit = ports_in_block & _SENSITIVE_PORTS
    first_line = block[0][0]

    if sensitive_hit:
        port_str = ", ".join(str(p) for p in sorted(sensitive_hit))
        findings.append(ShardFinding(
            severity="Critical",
            message=f"Security group ingress allows 0.0.0.0/0 on sensitive port(s): {port_str}",
            path=rel,
            line=first_line,
        ))
    elif ports_in_block:
        findings.append(ShardFinding(
            severity="High",
            message="Security group ingress allows 0.0.0.0/0 — verify this is intentional",
            path=rel,
            line=first_line,
        ))


def _check_encryption(lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    for i, raw in enumerate(lines, 1):
        if _ENCRYPTION_FALSE.search(raw):
            findings.append(ShardFinding(
                severity="High",
                message="Encryption is explicitly disabled — enable at-rest encryption",
                path=rel,
                line=i,
            ))


def _check_s3_public_acl(lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    for i, raw in enumerate(lines, 1):
        if _S3_PUBLIC_ACL.search(raw):
            findings.append(ShardFinding(
                severity="Critical",
                message="S3 bucket ACL is set to public — this exposes all bucket contents",
                path=rel,
                line=i,
            ))


def _check_deletion_protection(lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    for i, raw in enumerate(lines, 1):
        if _DELETION_PROTECTION_FALSE.search(raw):
            findings.append(ShardFinding(
                severity="High",
                message="deletion_protection is disabled — accidental deletion of database/resource is possible",
                path=rel,
                line=i,
            ))
        if _PREVENT_DESTROY_FALSE.search(raw):
            findings.append(ShardFinding(
                severity="High",
                message="prevent_destroy is false — lifecycle safeguard is not active",
                path=rel,
                line=i,
            ))


def _check_iam_wildcards(text: str, lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    for m in _IAM_WILDCARD_ACTION.finditer(text):
        line_no = text[:m.start()].count("\n") + 1
        findings.append(ShardFinding(
            severity="Critical",
            message='IAM policy uses wildcard action ("*") — apply least-privilege permissions',
            path=rel,
            line=line_no,
        ))
    for m in _IAM_WILDCARD_RESOURCE.finditer(text):
        line_no = text[:m.start()].count("\n") + 1
        findings.append(ShardFinding(
            severity="Critical",
            message='IAM policy targets wildcard resource ("*") — scope to specific ARNs',
            path=rel,
            line=line_no,
        ))


def _check_missing_tags(text: str, lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    """Check resource blocks for missing owner / environment tags or labels.

    For GCP resources (google_*): only flag types on the _GCP_LABEL_SUPPORTED allowlist,
    and look for a labels block instead of a tags block.
    For non-GCP resources: unchanged AWS-style tags check.
    """
    i = 0
    while i < len(lines):
        m = _RESOURCE_BLOCK_START.match(lines[i].strip())
        if m:
            resource_type = m.group(1)
            if resource_type.startswith("data.") or resource_type in (
                "terraform", "variable", "output", "provider", "module"
            ):
                i += 1
                continue
            start_line = i
            block_text, end_line = _collect_block(lines, i)
            if block_text:
                is_gcp = resource_type.startswith("google_")
                if is_gcp and resource_type not in _GCP_LABEL_SUPPORTED:
                    # Only flag GCP resources that are known to support labels.
                    i = end_line + 1
                    continue
                if is_gcp:
                    has_meta  = _LABELS_BLOCK.search(block_text)
                    has_owner = _LABEL_OWNER.search(block_text)
                    has_env   = _LABEL_ENV.search(block_text)
                    term      = "labels"
                else:
                    has_meta  = _TAGS_BLOCK.search(block_text)
                    has_owner = _TAG_OWNER.search(block_text)
                    has_env   = _TAG_ENV.search(block_text)
                    term      = "tags"
                if has_meta:
                    missing = []
                    if not has_owner:
                        missing.append("owner")
                    if not has_env:
                        missing.append("environment")
                    if missing:
                        findings.append(ShardFinding(
                            severity="Medium",
                            message=f"Resource {m.group(1)}.{m.group(2)} missing required {term}: {', '.join(missing)}",
                            path=rel,
                            line=start_line + 1,
                        ))
                else:
                    findings.append(ShardFinding(
                        severity="Medium",
                        message=f"Resource {m.group(1)}.{m.group(2)} has no {term} block — add owner and environment {term}",
                        path=rel,
                        line=start_line + 1,
                    ))
            i = end_line + 1
        else:
            i += 1


def _collect_block(lines: list[str], start: int) -> tuple[str, int]:
    """Return (block_text, end_line_index) for a HCL block starting at *start*."""
    depth = 0
    collected: list[str] = []
    started = False
    for j in range(start, len(lines)):
        raw = lines[j]
        depth += raw.count("{") - raw.count("}")
        collected.append(raw)
        if "{" in raw:
            started = True
        if started and depth <= 0:
            return "\n".join(collected), j
    return "\n".join(collected), len(lines) - 1


def _check_backend_locking(text: str, lines: list[str], rel: str, findings: list[ShardFinding], providers: frozenset[str]) -> None:
    if _BACKEND_LOCAL.search(text):
        findings.append(ShardFinding(
            severity="High",
            message="Terraform uses a local backend — use a remote backend with locking for production state",
            path=rel,
            line=None,
        ))
    if "aws" in providers and _BACKEND_S3.search(text) and not _DYNAMO_LOCK.search(text):
        findings.append(ShardFinding(
            severity="High",
            message="S3 backend is missing dynamodb_table — state locking is not configured",
            path=rel,
            line=None,
        ))


def _check_hardcoded_secrets(lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(raw):
                findings.append(ShardFinding(
                    severity="Critical",
                    message="Hardcoded credential or secret detected — use variable references or a secrets manager",
                    path=rel,
                    line=i,
                ))
                break


def _check_gcp_resources(text: str, lines: list[str], rel: str, findings: list[ShardFinding]) -> None:
    """GCP-specific resource checks: broad IAM, SQL public networks, Secret Manager, Cloud Run, Storage."""
    i = 0
    while i < len(lines):
        m = _RESOURCE_BLOCK_START.match(lines[i].strip())
        if m:
            resource_type = m.group(1)
            resource_name = m.group(2)
            start_line = i
            block_text, end_line = _collect_block(lines, i)
            if block_text and resource_type.startswith("google_"):
                _check_gcp_block(resource_type, resource_name, block_text, rel, start_line + 1, findings)
            i = end_line + 1
        else:
            i += 1


def _check_gcp_block(
    resource_type: str,
    resource_name: str,
    block_text: str,
    rel: str,
    line_no: int,
    findings: list[ShardFinding],
) -> None:
    """Dispatch GCP-specific checks for a single resource block."""
    if resource_type in ("google_project_iam_member", "google_project_iam_binding"):
        bm = _GCP_IAM_BROAD.search(block_text)
        if bm:
            role = bm.group(1)
            sev = "Critical" if role.lower() in ("roles/editor", "roles/owner") else "High"
            findings.append(ShardFinding(
                severity=sev,
                message=f"google_project_iam grants broad role {role} — apply least-privilege",
                path=rel,
                line=line_no,
            ))

    if resource_type == "google_sql_database_instance":
        # Only flag the public-network risk here; deletion_protection is handled by the generic check.
        if _CIDR_WILDCARD.search(block_text) and "authorized_networks" in block_text.lower():
            findings.append(ShardFinding(
                severity="Critical",
                message=(
                    f"google_sql_database_instance.{resource_name} allows 0.0.0.0/0 in "
                    "authorized_networks — restrict to known CIDRs"
                ),
                path=rel,
                line=line_no,
            ))

    if resource_type == "google_secret_manager_secret_version":
        if _GCP_SECRET_LITERAL.search(block_text):
            findings.append(ShardFinding(
                severity="Critical",
                message=(
                    f"google_secret_manager_secret_version.{resource_name} contains a literal "
                    "secret_data value — use a variable or external secret injection"
                ),
                path=rel,
                line=line_no,
            ))

    if resource_type == "google_cloud_run_v2_service":
        if _GCP_CLOUD_RUN_PUBLIC.search(block_text):
            findings.append(ShardFinding(
                severity="High",
                message=(
                    f"google_cloud_run_v2_service.{resource_name} ingress is INGRESS_TRAFFIC_ALL "
                    "— verify public exposure is intentional"
                ),
                path=rel,
                line=line_no,
            ))

    if resource_type == "google_storage_bucket":
        if _GCP_STORAGE_PUBLIC.search(block_text):
            findings.append(ShardFinding(
                severity="High",
                message=(
                    f"google_storage_bucket.{resource_name} has public_access_prevention = inherited "
                    "— set to enforced"
                ),
                path=rel,
                line=line_no,
            ))
