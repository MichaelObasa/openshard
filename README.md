# OpenShard

OpenShard is a local-first control layer for AI coding agents.

It gives developers a terminal workflow for asking, planning, running, inspecting, and recording AI-assisted code work - without blindly trusting a model or hiding what happened.

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CLI-terminal-black?style=for-the-badge" alt="CLI"></a>
</p>

---

## The loop

```
Ask → Plan → Run → Inspect → Feedback
```

**Ask** - Ask OpenShard product, model, and command questions.
**Plan** - Generate a local execution plan. Plan Mode v1 is deterministic and local: it does not scan the repo, call a provider, or write files.
**Run** - Send a real repo task through OpenShard's controlled execution path.
**Inspect** - Review the result, actions taken, evidence, checks, cost estimate, model choice, and Shard receipt.
**Feedback** - Record whether the result was accepted, partial, rejected, or needs more work.

---

## Why this exists

AI coding agents are powerful, but teams still need control: receipts, evidence, policy, cost awareness, and review boundaries.
OpenShard isn't another chat UI - it's the control and audit layer around agentic coding work.

---

## Quick demo

<p align="center">
  <img src="demos/Openshard_Final_Demo_Full.gif" alt="OpenShard Demo" width="800"/>
</p>

Launch the TUI:

```bash
openshard tui
```

Inside the TUI:

```
/ask what models do you support?
/plan review this repo for production readiness
Review this Terraform repo for production readiness. Focus on security, deletion protection, networking risk, secrets, and 2am operability. Do not apply changes.
/last more
/feedback accepted
```

CLI commands:

```bash
openshard models list
openshard models mode ask
openshard models mode plan
openshard last
openshard last --more
openshard session infer
```

<p align="center">
  <img src="demos/openshard_last_--more.gif" alt="OpenShard Last --more" width="800"/>
</p>

---

## What OpenShard records

Each run produces a **Shard receipt** containing:

- Model used and cost estimate
- Risk level
- Changed files and touched/evidence files
- Checks and their outcomes
- Evidence
- Actions timeline

OpenShard can also record local feedback and infer session signals around the run.

Raw developer content is not stored by default.

```bash
openshard last --more    # expanded receipt for the latest run
openshard last --full    # full stored details
```

---

## Workflow packs

Workflow packs are pre-built prompts for repeatable engineering reviews.

```bash
openshard packs list
openshard packs show production-iac-hardening
openshard packs prompt production-iac-hardening
```

Built-in packs include: `repo-explanation`, `production-iac-hardening`, `terraform-networking-review`, `iam-security-review`, `cicd-safety-review`, and `powershell-automation-review`.

---

## What works today

- Local CLI and TUI (`openshard tui`)
- Ask Mode - local, deterministic product and model Q&A
- Plan Mode v1 - local plan generation (not repo-aware, no provider calls)
- Controlled run path for real repo tasks
- Shard receipts with model, risk, files, checks, cost, and evidence
- `/last` and `/last --more` inside TUI and CLI
- 37-model registry with Ask/Plan model policy inspection
- Advisory-only model routing surfaces (model advisory, feedback advisory)
- Read-only review handling - preserves "Changed 0 files", no report files by default
- Basic intent-specific read-only review handling for Terraform/IaC, CI/CD, auth/security, tests, and docs/onboarding
- Feedback and session signal inference (`openshard session infer`)
- Workflow packs for repeatable engineering reviews

---

## What is not done yet

- Ask Mode and Plan Mode are local deterministic v1 - they do not call a provider
- Plan Mode is not repo-aware yet
- Feedback advisory does not automatically change routing yet
- No cloud or team dashboard yet
- No IDE integration yet
- No PyPI or Homebrew release yet - install from GitHub
- External testing is still ongoing

---

## Installation

**Recommended (pipx):**

```bash
pipx install git+https://github.com/MichaelObasa/openshard.git
```

Run:

```bash
openshard tui
```

**Alternative (uv):**

```bash
uv tool install git+https://github.com/MichaelObasa/openshard.git
```

**Local development:**

```bash
git clone https://github.com/MichaelObasa/openshard.git
cd openshard
pip install -e .
```

See [docs/install.md](docs/install.md) for upgrade instructions and notes.

---

## Production-infra demo

The `examples/production-infra-demo/` directory contains a fictional GCP workload called **DocuVault** - a sanitised demo scenario for OpenShard.

The infrastructure is intentionally production-shaped: networking, IAM, Cloud SQL, Cloud Run, storage, secrets, monitoring, and logging. It is **deliberately flawed** to serve as the input for an IaC hardening review.

All names, project IDs, resource IDs, CIDRs, and accounts are fake and public-safe. No employer or customer details. Designed to show a serious IaC review, not a toy example.

See [`examples/production-infra-demo/README.md`](examples/production-infra-demo/README.md) and [`examples/production-infra-demo/demo-task.md`](examples/production-infra-demo/demo-task.md).

---

## Developer setup

```bash
git clone https://github.com/MichaelObasa/openshard.git
cd openshard
pip install -e .
python -m pytest -q
python -m ruff check .
```

---

## Advanced: evals

OpenShard also includes a local eval harness for checking routing and workflow behaviour.

```bash
openshard eval list
openshard eval validate --suite basic
```

---

## Why Open Source?

Routing decisions should be inspectable. If a tool decides which model touches security-sensitive code, developers should be able to see why.

OpenShard is open because trust, integrations, and routing policies improve when real users can inspect and extend the system.

---

## Contributing

Contributions are welcome around:

- routing policies and scoring logic,
- repo analyzers for new stacks,
- model profiles and capability data,
- evaluation datasets,
- provider integrations,
- CLI UX improvements.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Security

If you find a security issue, please report it privately before opening a public issue. See [SECURITY.md](SECURITY.md).

---

## License

MIT
