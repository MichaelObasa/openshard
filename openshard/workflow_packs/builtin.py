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
    {
        "id": "code-review",
        "title": "Code review",
        "category": "code_quality",
        "summary": "Reviews recent code changes for bugs, logic errors, unclear naming, and missed edge cases.",
        "prompt": (
            "Review the recent code changes in this repository. Look for logic bugs, off-by-one errors, "
            "unhandled edge cases, unclear variable or function names, and any code that is harder to read "
            "than it needs to be. Call out specific lines or functions where the issue lives. Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo after staging or committing changes you want reviewed.",
        "expected_receipt_value": "A list of specific issues with file and line references, covering bugs, edge cases, and clarity problems.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["code", "review", "quality"],
    },
    {
        "id": "test-coverage-gaps",
        "title": "Test coverage gaps",
        "category": "testing",
        "summary": "Identifies untested code paths, missing edge cases, and weak assertions across the test suite.",
        "prompt": (
            "Analyse the test suite in this repository and identify code paths that have no test coverage, "
            "edge cases that are not exercised (empty input, boundary values, error paths), and tests that "
            "make weak or meaningless assertions. For each gap, name the untested function or branch and "
            "explain what a good test would check. Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo. Works best when the test directory is visible alongside source files.",
        "expected_receipt_value": "A prioritised list of coverage gaps with the untested function or branch named and a suggested assertion for each.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["testing", "coverage", "quality"],
    },
    {
        "id": "dependency-audit",
        "title": "Dependency audit",
        "category": "security",
        "summary": "Audits project dependencies for known vulnerabilities, severely outdated packages, and supply chain risks.",
        "prompt": (
            "Audit the dependencies declared in this project (package.json, requirements.txt, go.mod, Gemfile, "
            "or equivalent). Identify packages with known CVEs, packages that are more than two major versions "
            "behind the current release, packages with unusual install scripts or broad filesystem access, and "
            "any transitive dependencies that are pinned to a suspicious commit or mirror. Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo containing the dependency manifest files.",
        "expected_receipt_value": "A list of risky dependencies grouped by concern: known CVEs, severely outdated, and supply chain red flags.",
        "safety_notes": "Read-only. Does not modify files or run install commands.",
        "tags": ["dependencies", "security", "supply-chain"],
    },
    {
        "id": "api-design-review",
        "title": "API design review",
        "category": "code_quality",
        "summary": "Reviews REST, GraphQL, or gRPC API definitions for consistency, versioning strategy, error handling, and security gaps.",
        "prompt": (
            "Review the API definitions in this codebase—routes, schemas, proto files, or OpenAPI specs. "
            "Identify inconsistent naming conventions (mixed snake_case and camelCase, inconsistent pluralisation), "
            "missing or broken versioning strategy, error responses that leak internal details, endpoints that "
            "lack authentication or authorisation checks, and any breaking changes made without a version bump. "
            "Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo. Works best when route definitions, schema files, or OpenAPI specs are present.",
        "expected_receipt_value": "A list of API design issues grouped by type: consistency, versioning, error handling, and security.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["api", "design", "review"],
    },
    {
        "id": "docker-security-review",
        "title": "Docker security review",
        "category": "infrastructure",
        "summary": "Reviews Dockerfiles and docker-compose files for root user usage, exposed secrets, unpinned images, and unnecessary capabilities.",
        "prompt": (
            "Review all Dockerfiles and docker-compose files in this repository. Check for containers running "
            "as root, image tags pinned to 'latest' or unpinned entirely, secrets or credentials passed via "
            "ENV or ARG, unnecessary capabilities or privileged mode, ports exposed wider than needed, and "
            "base images with known vulnerabilities or untrusted registries. Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from the root of the repo containing Dockerfiles or docker-compose.yml.",
        "expected_receipt_value": "A list of Docker security issues grouped by severity, covering image pinning, user context, secrets, and capabilities, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No Docker files modified.",
        "tags": ["docker", "containers", "security", "infrastructure"],
    },
    {
        "id": "readme-audit",
        "title": "README audit",
        "category": "documentation",
        "summary": "Audits README and project docs for accuracy, missing setup steps, outdated commands, and onboarding gaps.",
        "prompt": (
            "Read the README and any other documentation files in this repository. Check for setup instructions "
            "that are incomplete or reference commands that do not match the actual project structure, "
            "prerequisites that are assumed but never stated, outdated dependency versions or tool names, "
            "and sections that describe features no longer present in the code. Do not modify files."
        ),
        "recommended_context": "Run from the root of the repo. Works best when README.md and any docs/ directory are present.",
        "expected_receipt_value": "A list of documentation issues covering missing steps, inaccurate commands, outdated content, and onboarding gaps.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["documentation", "readme", "onboarding"],
    },
    {
        "id": "database-migration-review",
        "title": "Database migration review",
        "category": "delivery",
        "summary": "Reviews SQL migration files for irreversible operations, missing transactions, table-locking risks, data-loss hazards, and missing index coverage.",
        "prompt": (
            "Review all database migration files in this directory (SQL files, Alembic versions, Flyway scripts, "
            "Liquibase changelogs, or similar). For each migration, identify: operations that cannot be rolled back "
            "(DROP COLUMN, DROP TABLE, destructive ALTER) without a corresponding down migration; statements that run "
            "outside an explicit transaction and could leave the schema in a partial state on failure; DDL operations "
            "on large tables that will take an exclusive lock and block production traffic; direct data deletions or "
            "type-narrowing conversions that risk data loss; and new foreign keys or query patterns that will hit "
            "unindexed columns. Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "recommended_context": "Run from the directory containing migration files (.sql, Alembic versions/, Flyway sql/, Liquibase changelogs/, or similar).",
        "expected_receipt_value": "A list of migration risks grouped by severity: irreversible operations, transaction safety, locking hazards, data loss, and missing indexes, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No migration files modified.",
        "tags": ["database", "migrations", "sql", "safety"],
    },
    {
        "id": "accessibility-audit",
        "title": "Accessibility audit",
        "category": "code_quality",
        "summary": "Reviews frontend source files for accessibility issues including missing alt text, keyboard navigation gaps, missing ARIA roles, and form labelling problems.",
        "prompt": (
            "Review the frontend source files in this directory (HTML, JSX, TSX, Vue, Svelte, or similar). "
            "Identify: images and icon buttons missing meaningful alt text or aria-label; interactive elements "
            "that are not reachable or activatable via keyboard alone (missing tabIndex, no focus styles, click "
            "handlers on non-interactive elements); missing or incorrect ARIA roles, aria-expanded, aria-controls, "
            "and aria-live regions where dynamic content changes; form inputs that lack an associated label element "
            "or aria-labelledby; and colour-contrast issues where text or interactive elements use foreground/background "
            "combinations below WCAG AA thresholds. Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "recommended_context": "Run from the frontend source directory containing HTML, JSX, TSX, Vue, or Svelte files.",
        "expected_receipt_value": "A list of accessibility violations grouped by type: alt text, keyboard access, ARIA, form labels, and colour contrast, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No source files modified.",
        "tags": ["accessibility", "a11y", "frontend", "html"],
    },
    {
        "id": "kubernetes-security-review",
        "title": "Kubernetes security review",
        "category": "infrastructure",
        "summary": "Reviews Kubernetes manifests for containers running as root, missing resource limits, overly broad RBAC, exposed secrets, and missing network policies.",
        "prompt": (
            "Review all Kubernetes manifest files in this directory (Deployment, StatefulSet, DaemonSet, Job, "
            "CronJob, Role, ClusterRole, RoleBinding, NetworkPolicy, and related resources). Identify: containers "
            "that run as root or do not set securityContext.runAsNonRoot; containers or pods missing CPU and memory "
            "resource limits; containers with privileged: true or allowPrivilegeEscalation: true; Role and "
            "ClusterRole bindings that grant wildcard verbs or resources, or bind to system:masters; environment "
            "variables or volume mounts that expose Kubernetes Secrets in plaintext; and namespaces or workloads "
            "with no NetworkPolicy restricting ingress or egress. Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "workflow": "native",
        "recommended_context": "Run from the directory containing Kubernetes YAML manifests.",
        "expected_receipt_value": "A list of Kubernetes security issues grouped by severity: root containers, missing limits, privilege escalation, RBAC, secret exposure, and network policies, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No manifest files modified.",
        "tags": ["kubernetes", "k8s", "security", "infrastructure"],
    },
    {
        "id": "openapi-spec-review",
        "title": "OpenAPI spec review",
        "category": "code_quality",
        "summary": "Reviews OpenAPI/Swagger specs for missing response codes, undocumented errors, inconsistent naming, missing authentication schemes, and fields without descriptions.",
        "prompt": (
            "Review the OpenAPI or Swagger specification files in this directory (openapi.yaml, swagger.json, "
            "or similar). Identify: operations missing standard HTTP error response codes (400, 401, 403, 404, "
            "500); error responses that do not document their schema or response body; path and parameter names "
            "that mix naming conventions (camelCase vs snake_case, inconsistent plural/singular resources); "
            "operations that are accessible without any securityScheme or security requirement defined; and "
            "schema properties, parameters, or request body fields that lack a description or at least one example. "
            "Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "recommended_context": "Run from the directory containing openapi.yaml, swagger.json, or similar spec files.",
        "expected_receipt_value": "A list of spec issues grouped by type: missing response codes, undocumented errors, naming inconsistencies, auth gaps, and missing descriptions, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No spec files modified.",
        "tags": ["openapi", "swagger", "api", "documentation"],
    },
    {
        "id": "github-actions-review",
        "title": "GitHub Actions review",
        "category": "delivery",
        "summary": "Reviews GitHub Actions workflows for unpinned action versions, overly broad permissions, unsafe secret handling, unguarded production deploys, and missing timeouts.",
        "prompt": (
            "Review all workflow files in the .github/workflows directory. Identify: actions referenced by a "
            "mutable tag (v1, main, latest) rather than a pinned commit SHA; workflows or jobs that set "
            "permissions: write-all or inherit broad default permissions without restricting to only what is "
            "needed; steps that print, log, or expose secret values via run: echo or set-output; deploy jobs "
            "that push to production environments without a required approval gate or environment protection rule; "
            "and jobs or steps that have no timeout-minutes set and could run indefinitely on a hung process. "
            "Do not modify files."
        ),
        "execution_prompt_suffix": _EXECUTION_FINDINGS_SUFFIX,
        "recommended_context": "Run from the .github/workflows directory.",
        "expected_receipt_value": "A list of workflow risks grouped by type: unpinned actions, permissions, secret handling, deploy gates, and missing timeouts, plus a STRUCTURED_FINDINGS line.",
        "safety_notes": "Read-only analysis. No workflow files modified.",
        "tags": ["github-actions", "ci", "delivery", "security"],
    },
    {
        "id": "performance-hotspots",
        "title": "Performance hotspots",
        "category": "code_quality",
        "summary": "Identifies performance hotspots including N+1 query patterns, missing database indexes, unnecessary re-renders, synchronous I/O in async contexts, and unbounded data fetches.",
        "prompt": (
            "Analyse the source code in this directory for performance hotspots. Look for: ORM query patterns "
            "inside loops that produce N+1 database queries, especially in list views or serialisers; model "
            "relationships or filter conditions that traverse columns with no database index; React or Vue "
            "components that trigger expensive re-renders on every parent update due to inline object or function "
            "creation in render; synchronous file or network I/O calls made inside async route handlers or "
            "event loops where a non-blocking equivalent exists; and query or API call sites that fetch all rows "
            "or all fields with no pagination, limit clause, or field selection. For each hotspot, name the "
            "file and function where the issue lives and describe the specific fix. Do not modify files."
        ),
        "recommended_context": "Run from the source directory of the application.",
        "expected_receipt_value": "A list of performance hotspots by type: N+1 queries, missing indexes, unnecessary re-renders, blocking I/O, and unbounded fetches, each with file and function references.",
        "safety_notes": "Read-only. Does not modify files.",
        "tags": ["performance", "optimisation", "database", "frontend"],
    },
]
