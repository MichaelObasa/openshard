# OpenShard

<p align="center">
  <strong>The neutral control layer for AI coding workflows.</strong>
</p>

<p align="center">
  Route tasks across models and agents. Gate risky actions. Verify the result. Keep an execution record.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CLI-terminal-black?style=for-the-badge" alt="CLI"></a>
</p>

---

OpenShard is a CLI-first developer tool that sits above model APIs and coding workflows. It inspects a repo, routes the task to the right model or workflow, applies approval gates, can run verification, and records what happened.

Most coding tools make you choose the model, decide when to approve writes, remember what happened, and manually compare whether a cheaper or stronger model would have done better. OpenShard turns that into an explicit control layer:

- **Route** tasks by risk, complexity, repo context, model capability, and cost.
- **Gate** high-cost, risky-path, unsafe-command, and stack-mismatch actions.
- **Verify** generated changes when verification is enabled, using detected or configured test commands.
- **Record** model choice, cost, duration, changed files, verification status, retries, profiles, and matched skills.
- **Adjust routing optionally** from local run history and eval results when `--history-scoring` or `--eval-scoring` is enabled.

OpenShard is not trying to replace every coding agent. It is the layer above them: the place where routing, policy, verification, and execution records become inspectable.

---

## Demo

### Running a task

<p align="center">
  <img src="demos/Openshard_Final_Demo_Full.gif" alt="OpenShard Demo" width="800"/>
</p>

### Inspecting the result

<p align="center">
  <img src="demos/openshard_last_--more.gif" alt="OpenShard Last --more" width="800"/>
</p>

---

## Install

```bash
git clone https://github.com/MichaelObasa/openshard.git
cd openshard
pip install -e .
```

Set a provider key. OpenRouter is the default provider:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Direct Anthropic and OpenAI providers are also available:

```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

Verify the CLI:

```bash
openshard --help
```

---

## Quick Start

Plan a task:

```bash
openshard plan "build a FastAPI CRUD service with tests"
```

Run it and allow file writes:

```bash
openshard run "build a FastAPI CRUD service with tests" --write
```

Review what happened:

```bash
openshard last --more
openshard metrics
```

For safer execution, ask for an explicit plan before running and verify after writing:

```bash
openshard run "add auth middleware with tests" --write --verify --plan
```

---

## Commands

Core workflow:

```bash
openshard plan "<task>"          # produce a structured execution plan
openshard run "<task>"           # route and execute a task
openshard last                   # show the most recent run
openshard last --more            # include files, routing, profile, tokens, and notes
openshard last --full            # include full stored details
openshard metrics                # aggregate run-history metrics
openshard report                 # summarize recent executions
openshard explain "<task>"       # explain routing/category choice
```

Model, profile, and skill history:

```bash
openshard models                 # list cached models and capabilities
openshard models --refresh       # refresh provider model metadata
openshard models stats           # per-model performance from run history
openshard profiles stats         # per-profile cost, retry, and pass-rate metrics
openshard skills stats           # per-skill usage and performance metrics
```

Eval harness:

```bash
openshard eval list              # list eval tasks in the default suite
openshard eval validate          # validate eval task fixtures
openshard eval run               # run the default eval suite
openshard eval report            # summarize recorded eval runs
openshard eval compare --models modelA,modelB
openshard eval stats             # grouped pass/fail stats from eval history
```

Important `run` flags:

- `--write` writes generated files to disk.
- `--verify` runs verification after writing and enables retry on verification failure.
- `--dry-run` previews generated changes without writing.
- `--plan` shows an execution plan and prompts before running.
- `--workflow [auto|direct|staged|opencode]` chooses the execution workflow.
- `--approval [auto|smart|ask]` controls approval gates.
- `--provider [openrouter|anthropic|openai]` overrides the API provider.
- `--history-scoring` applies local run-history bonuses or penalties to model scoring.
- `--eval-scoring` applies local eval-result bonuses or penalties to model scoring.
- `--more` and `--full` expand run output.

`--executor` is deprecated. Use `--workflow` instead.

---

## Workflows

OpenShard separates routing decisions from execution shape.

| Workflow | Status | Notes |
|---|---:|---|
| `auto` | available | Selects a workflow from task category, repo risk, verification setting, and optional history signals. |
| `direct` | available | Uses a single model call through the configured provider. |
| `staged` | available | Uses a planning stage followed by an implementation stage through direct execution. |
| `opencode` | experimental | Delegates execution to the local OpenCode CLI. Requires `opencode` on PATH. |
| `native` | reserved | Future OpenShard Native direction; not shipped as a workflow. |
| `claude-code` | reserved | Defined in CLI choices but currently unavailable. |
| `codex` | reserved | Defined in CLI choices but currently unavailable. |

Execution profiles add another signal for how much effort to spend:

- `native_light` is selected for simpler, lower-risk tasks.
- `native_deep` is selected for security, complex, risky-path, or long tasks.
- `native_swarm` is an explicit experimental profile and is never auto-selected.

---

## Safety And Verification

OpenShard is built around the idea that model output should be controlled, not blindly trusted.

Approval gates can prompt before:

- writing files,
- touching risky paths,
- running unknown or unsafe shell commands,
- accepting high estimated cost,
- applying changes that do not match the detected repo stack.

Verification uses a configured command when present, or a detected command such as `python -m pytest` or `npm test` when available. Commands are classified before execution, and safe verification runs without invoking a shell.

When `--verify` is enabled and verification fails, OpenShard can retry generation with a stronger model. General API or generation failures are surfaced as errors with details available in `openshard last --more` or `openshard last --full`.

---

## Routing Evidence

OpenShard starts with task and repo signals:

- risk areas such as auth, payments, migrations, and security-sensitive files,
- task complexity and likely verification strength,
- detected languages, package files, framework, tests, risky paths, and changed files,
- provider inventory, model capabilities, context window, and cost metadata.

Local evidence is opt-in:

- `--history-scoring` uses previous run outcomes, cost, duration, retries, and verification status.
- `--eval-scoring` uses recorded eval outcomes from `.openshard/eval-runs.jsonl`.

This keeps the default behavior simple while allowing teams to route from evidence collected in their own repos.

---

## Evals

The bundled eval harness gives you small, repeatable tasks for checking model behavior.

```bash
openshard eval list --suite basic
openshard eval validate --suite basic
openshard eval run --suite basic --model anthropic/claude-haiku-4-5-20251001
openshard eval compare --suite basic --models modelA,modelB
openshard eval report
openshard eval stats
```

Eval runs write structured records under `.openshard/eval-runs.jsonl`. Reports show pass/fail rates, duration, token averages when available, and unsafe file attempts. Those results can become routing signals when `--eval-scoring` is enabled.

---

## Local Skills

Skills let you add repo-local guidance without changing OpenShard code.

```text
.openshard/skills/<slug>/SKILL.md
```

Each skill can define frontmatter such as:

| Field | Required | Description |
|---|---:|---|
| `name` | yes | Human-readable skill name shown in logs. |
| `description` | no | One-line summary. |
| `category` | no | Broad category such as `security`, `infrastructure`, or `performance`. |
| `keywords` | no | Terms matched against the task text. |
| `languages` | no | Languages such as `python` or `typescript`. |
| `framework` | no | Framework matched against detected repo stack. |

A skill can match by category, keyword, framework, or language plus category. Up to five matched skills are included per run. Skill names, categories, descriptions, and match reasons are surfaced as prompt context; scripts inside `SKILL.md` are never executed.

See [`examples/skills/`](examples/skills/) for starters.

---

## Configuration

OpenShard reads `config.yml` from the repo root. Minimal example matching the public config shape:

```yaml
model_tiers:
  - name: fast
    model: anthropic/claude-haiku-4.5
    max_tokens: 1024
  - name: balanced
    model: anthropic/claude-sonnet-4.6
    max_tokens: 4096
  - name: powerful
    model: anthropic/claude-opus-4.6
    max_tokens: 8192

planning_model: anthropic/claude-sonnet-4.6
execution_model: anthropic/claude-sonnet-4.6
fixer_model: anthropic/claude-sonnet-4.6

workflow: auto
approval_mode: smart
cost_gate_threshold: 0.10
```

You can also set `verification_command` in config when auto-detection is not enough.

---

## Where It Fits

OpenShard sits above model APIs and coding workflows:

| Layer | Examples |
|---|---|
| Models | OpenAI, Anthropic, and models exposed through OpenRouter |
| Access | OpenRouter or direct provider keys |
| Execution | Direct provider calls today; experimental OpenCode delegation |
| Control | OpenShard routing, gates, verification, history, evals, and metrics |

Use OpenShard when you want a local, inspectable CLI for model choice, approval policy, verification, and run records across real repo work.

It is less useful if you only need occasional one-off chat edits, do not care which model handles a task, or do not want a CLI-first workflow.

---

## Roadmap

High-level areas under active exploration:

- more task-aware routing from eval and run-history evidence,
- expanded eval suites and cost-per-pass reporting,
- native execution improvements,
- deeper integrations with OpenCode, Claude Code, and Codex,
- team policies and hosted dashboards,
- longer-term agent control layer with orchestration, permissions, audit trails, and observability.

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
