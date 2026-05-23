from __future__ import annotations

# Internal execution suffix: appended to the prompt sent to the model for review packs.
# Never shown in the TUI composer, CLI prompt display, or any user-facing output.
_EXECUTION_FINDINGS_SUFFIX = (
    "\n\nAfter completing your analysis, put the following content in the JSON `summary` field "
    "of your response (multi-line content is allowed for review tasks):\n\n"
    "Found X issues worth addressing.\n\n"
    "Critical\n"
    "- <one-sentence finding>\n\n"
    "High\n"
    "- <one-sentence finding>\n\n"
    "Only include severity levels that have findings. "
    "Use 'No issues found.' if the review is clean.\n\n"
    "End the summary field with this exact line (do not wrap it in a code block):\n"
    "STRUCTURED_FINDINGS: [{\"severity\": \"Critical\", \"message\": \"One sentence finding\"}, "
    "{\"severity\": \"High\", \"message\": \"Another finding\"}]\n"
    "Severity must be one of: Critical, High, Medium, Low. "
    "Include every significant finding. This line is required for structured reporting."
)

_BUILTIN_PACKS: list[dict] = [
    {
        "id": "repo-explanation",
        "title": "Explain this repo",
        "category": "repo_review",
        "summary": "Produces a plain-English overview of an unfamiliar codebase: main modules, entry points, tests, and risky areas.",
        "prompt": (
            "Explain this repo in plain English. Identify the main modules, likely entry points, tests, "
            "and any risky or unclear areas. Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo you want explained.",
        "expected_receipt_value": "A structured explanation covering modules, entry points, test coverage, and risk areas.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["repo", "onboarding", "review", "read-only"],
    },
    {
        "id": "production-iac-hardening",
        "title": "Production IaC hardening review",
        "category": "infrastructure",
        "summary": "Hardens a Terraform codebase by assessing security posture, operability at 2am, and developer experience for a small engineering team.",
        "prompt": (
            "Review and harden this deliberately flawed Terraform codebase. Assess it through security/compliance "
            "posture, 2am operability, and developer experience for a 5-10 person engineering team. Identify "
            "critical, high, and medium risks. Explain trade-offs. Do not apply changes directly without review."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from a directory containing Terraform files (.tf). Works best with the OpenShard production-infra-demo fixture.",
        "expected_receipt_value": "A prioritised list of critical, high, and medium risks with trade-off explanations, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Review-only by default. Do not apply changes without explicit confirmation.",
        "tags": ["terraform", "iac", "security", "compliance", "infrastructure"],
    },
    {
        "id": "terraform-networking-review",
        "title": "Terraform networking review",
        "category": "infrastructure",
        "summary": "Identifies risky firewall rules, overly broad ingress, missing egress controls, NAT/subnet issues, and unsafe defaults in Terraform networking config.",
        "prompt": (
            "Review this Terraform networking configuration and identify risky firewall rules, overly broad ingress, "
            "missing egress controls, NAT/subnet issues, and unsafe defaults before apply."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from a directory containing Terraform networking files (network.tf, firewall.tf, vpc.tf, or similar).",
        "expected_receipt_value": "A list of networking risks grouped by severity with remediation notes, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No changes applied.",
        "tags": ["terraform", "networking", "firewall", "iac", "infrastructure"],
    },
    {
        "id": "iam-security-review",
        "title": "IAM security review",
        "category": "security",
        "summary": "Reviews IAM configuration for privilege escalation risks, overly broad roles, wildcard permissions, missing conditions, and least-privilege gaps.",
        "prompt": (
            "Review this IAM configuration and identify privilege escalation risks, overly broad roles, "
            "wildcard permissions, missing conditions, and least-privilege gaps."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from a directory containing IAM configuration files (iam.tf, *.json policy files, or similar).",
        "expected_receipt_value": "A prioritised list of IAM risks with specific permissions or roles called out, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No IAM changes applied.",
        "tags": ["iam", "security", "permissions", "least-privilege"],
    },
    {
        "id": "cicd-safety-review",
        "title": "CI/CD pipeline safety review",
        "category": "delivery",
        "summary": "Reviews CI/CD pipeline configuration for unguarded production deploys, missing approval gates, exposed secrets, unsafe environment handling, and rollback gaps.",
        "prompt": (
            "Review this CI/CD pipeline and identify unguarded production deploys, missing approval gates, "
            "exposed secrets, unsafe environment handling, and rollback gaps."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from a directory containing CI/CD config files (.github/workflows, .gitlab-ci.yml, Jenkinsfile, or similar).",
        "expected_receipt_value": "A list of pipeline risks grouped by severity: deploy safety, secrets handling, and rollback readiness, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No pipeline config changes applied.",
        "tags": ["cicd", "pipeline", "delivery", "secrets", "security"],
    },
    {
        "id": "powershell-automation-review",
        "title": "PowerShell automation review",
        "category": "automation",
        "summary": "Reviews PowerShell deployment scripts for missing validation, poor error handling, secret-handling risks, idempotency issues, and rollback gaps.",
        "prompt": (
            "Review this PowerShell deployment script and identify missing validation, poor error handling, "
            "secret-handling risks, idempotency issues, and rollback gaps."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from a directory containing PowerShell scripts (.ps1 files).",
        "expected_receipt_value": "A list of script risks covering validation, error handling, secrets, idempotency, and rollback, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No script changes applied.",
        "tags": ["powershell", "automation", "scripting", "deployment"],
    },
]
