# OpenShard

<p align="center">
  <strong>A local-first control layer and native coding agent for AI-assisted engineering work.</strong>
</p>

<p align="center">
  AI coding agents can write code, but serious developers still need control: what ran, what changed, what context was used, whether checks passed, what it cost, and whether there is a receipt. OpenShard wraps AI coding work with routing, sandboxing, checks, receipts, findings, and feedback.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CLI-terminal-black?style=for-the-badge" alt="CLI"></a>
</p>

---

## What OpenShard does

- Runs AI-assisted engineering tasks through OpenShard Native
- Routes work to the right model where available (Model: Auto)
- Records each run as a Shard receipt
- Shows context provenance, files touched, checks, cost, findings, and feedback
- Supports workflow packs for repeatable engineering reviews
- Runs locally first

---

## Quick install

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

## 60-second demo

<p align="center">
  <img src="demos/Openshard_Final_Demo_Full.gif" alt="OpenShard Demo" width="800"/>
</p>

Launch the TUI:

```bash
openshard tui
```

Inside the TUI, browse and load a workflow pack:

```
/packs
/pack production-iac-hardening
```

Copy the prompt and run it, or run directly from the shell:

```bash
openshard run "Review and harden this deliberately flawed Terraform codebase..."
```

Inspect the Shard receipt for the latest run:

```bash
openshard last --more
```

<p align="center">
  <img src="demos/openshard_last_--more.gif" alt="OpenShard Last --more" width="800"/>
</p>

Leave feedback:

```bash
openshard feedback --outcome accepted --note "Useful review"
```

---

## Production-infra demo

The `examples/production-infra-demo/` directory contains a fictional GCP workload called **DocuVault** — a sanitised demo scenario for OpenShard.

The infrastructure is intentionally production-shaped: networking, IAM, Cloud SQL, Cloud Run, storage, secrets, monitoring, and logging. It is **deliberately flawed** to serve as the input for an IaC hardening review.

All names, project IDs, resource IDs, CIDRs, and accounts are fake and public-safe. No employer or customer details. Designed to show a serious IaC review, not a toy example.

See [`examples/production-infra-demo/README.md`](examples/production-infra-demo/README.md) and [`examples/production-infra-demo/demo-task.md`](examples/production-infra-demo/demo-task.md).

---

## Shard receipts

A Shard is the durable receipt for an AI engineering run. A Shard can show:

- Task and agent
- Model(s) used
- Strategy
- Context provenance
- Inspected files
- Changed and touched files
- Checks and their outcomes
- Findings (when structured findings exist)
- Cost
- Feedback

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

## What is built today

- OpenShard Native agent/harness
- TUI (`openshard tui`)
- Workflow packs
- Shard receipts with full run metadata
- Findings rendering when structured findings exist
- Feedback signals (`openshard feedback`)
- Context provenance
- Sandbox, diff, apply, and receipts where implemented
- Local-first run history

---

## What is not built yet

- No hosted team platform yet
- Not a full Claude Code or Codex replacement
- External harness adapters (OpenCode and others) are experimental, not guaranteed
- No PyPI or Homebrew release yet — install from GitHub
- No cloud sync yet

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
