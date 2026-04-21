# OpenShard

<p align="center">
  <strong>The routing layer for AI coding.</strong>
</p>

<p align="center">
  OpenShard routes your coding tasks to the right model for each stage. 
  Better results, lower cost.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CLI-terminal-black?style=for-the-badge" alt="CLI"></a>
</p>

---

## Why routing matters now

Most coding tasks aren't uniformly hard. Think about what a typical feature 
actually involves:

- **Architecture decisions** - high risk, needs strong reasoning
- **Business logic** - moderate complexity
- **Scaffolding / boilerplate** - mostly pattern matching
- **Tests** - structured generation
- **Refactoring** - consistency and correctness

Only a fraction of that genuinely needs the best model available. The rest just 
needs something good enough.

But developers are still doing the routing themselves - manually switching 
models, retrying failures, deciding what's worth frontier-model spend. 
There's no clean open-source solution for that.

That's what OpenShard is.

---

## What it does

You give OpenShard one engineering task.

It inspects your repo, breaks the work into stages, picks the right model for 
each stage based on difficulty, cost, and context - then runs it. You approve 
before it executes. You review when it's done.

```bash
openshard run "add email/password auth with protected routes and tests"
```

```
Inspecting repo...
Stack detected: Next.js + TypeScript + PostgreSQL

Plan:
  Auth design          ->  strong model      (security-sensitive)
  UI scaffolding       ->  cheaper model     (low-risk boilerplate)
  Test generation      ->  mid model         (generation-heavy)
  Final review         ->  strong model      (verification)

Estimated cost: $1.82
Proceed? [y/N]
```
✅ Task complete

Auth flow implemented
Protected routes added
Tests: 18/18 passing
No unresolved errors

Time: 2m 14s  |  Cost: $1.82  |  Details: openshard report

That is the point. Your task gets done. You don't think about models.

---

## Where OpenShard fits

OpenShard does not replace your tools. It sits above them.

| Layer | Tool |
|---|---|
| Models | OpenAI, Anthropic, Google, open-source |
| Access | OpenRouter or direct API keys |
| Execution | Claude Code, Codex, OpenCode or direct patching |
| **Routing** | **OpenShard** |

Think of it as the decision layer - the brain that decides which model does 
what, so you don't have to.

**Best used with OpenRouter** - it gives OpenShard a bigger choice of models 
with live cost metadata across providers.

---

## Quick start

**1. Install**
```bash
pip install openshard
```

**2. Set your provider key**
```bash
export OPENROUTER_API_KEY=your_key_here
```

**3. Plan before you run**
```bash
openshard plan "build a FastAPI CRUD service with tests"
```

**4. Run**
```bash
openshard run "build a FastAPI CRUD service with tests"
```

---

## Commands

```bash
openshard run "<task>"       # run the full workflow
openshard plan "<task>"      # see the plan before executing
openshard explain "<task>"   # understand the routing decisions
openshard report             # cost, time, and outcome summary
openshard models             # list available models
openshard config init        # set up your config
openshard doctor             # check your setup
```

---

## How it works

OpenShard has five core parts working in sequence:

**1. Task interpreter** - understands what you're asking  
**2. Repo analyzer** - detects stack, structure, and risky areas  
**3. Task splitter** - breaks work into stages by difficulty  
**4. Routing engine** - picks the right model per stage based on risk, 
cost, and capability  
**5. Executor** - applies patches, runs commands, verifies output, 
retries if needed
task -> inspect repo -> split stages -> route models -> execute -> verify -> report

---

## Routing logic

OpenShard uses a scoring approach, not a fixed model list. Each stage is 
scored on:

- **Risk** - auth, payments, migrations, security
- **Complexity** - novel logic vs. repetitive patterns
- **Verification strength** - can tests catch mistakes here?
- **Cost sensitivity** - where strong-model budget matters most

Then it maps to model classes:

| Stage type | Model class |
|---|---|
| Architecture / security-sensitive | Strongest available |
| Standard app logic | Mid-tier |
| Boilerplate / scaffolding | Cheapest capable |
| Tests / docs | Cost-efficient |
| Final review | Strongest available |

The model list is not hardcoded. Routing adapts as new models emerge.

---

## Configuration

> Model references below are accurate at time of release. The AI landscape 
moves fast - swap in whatever models suit your needs.

```yaml
provider: openrouter

workflow:
  - plan
  - execute
  - test
  - review

routing:
  planning:
    preferred_models:
      - anthropic/claude-sonnet-4-6      # strong reasoning
      - openai/gpt-5.4-thinking          # strong reasoning alternative
  hard_implementation:
    preferred_models:
      - anthropic/claude-sonnet-4-6      # strong coding
      - openai/gpt-5.4-thinking          # strong coding alternative
      - anthropic/claude-opus-4-7        # fallback/escalatiob
  cheap_execution:
    preferred_models:
      - google/gemini-3.1-pro            # capable, cost-efficient
      - google/gemma-4-31b               # open-source, low cost
      - zhipu/glm-5-1                    # open-source, low cost
      - moonshot/kimi-k2-6               # open-source, low cost
      - minimax/minimax-2-7              # open-source, low cost
      - deepseek/deepseek-v3-2           # open-source, low cost
  review:
    preferred_models:
      - anthropic/claude-opus-4-6        # strong review
      - openai/gpt-5.4-thinking          # strong review alternative
```

---

## When not to use OpenShard

OpenShard is not the right tool if:

- You only use Cursor or Claude Code casually for small tasks
- You're on a flat subscription and don't care about per-token cost
- You want a GUI-first experience
- You're not using APIs or OpenRouter

It works best for developers using APIs directly, running multi-step feature 
work, or anyone where cost, reliability, and consistency across a real repo 
actually matters.

---

## Status and roadmap

Alpha. The core CLI, routing engine, and OpenRouter integration are the 
first priority.

**v0.1 - current focus**
- CLI
- OpenRouter support
- Basic routing rules
- Plan / run / review flow
- Cost + time reporting

**What's coming**
- Repo-aware task splitting
- Cost estimation and baselines
- Retry and escalation logic
- Run history
- Local evals and policy tuning
- Integrations with Claude Code, Codex, Cline
- Adaptive routing from usage data
- Team policies and dashboards
- Cloud control plane
- IDE extension

---

## Why open source?

Routing decisions should be inspectable. If a tool is deciding which model 
touches your security-sensitive code, you should be able to see why.

OpenShard is open because trust matters, integrations matter, and the best 
routing policies will come from genuine usage from many users and not just 
one person's assumptions.

---

## Contributing

Contributions welcome around:

- Routing policies and scoring logic
- Repo analyzers for new stacks
- Model profiles and capability data
- Evaluation datasets
- Provider integrations
- CLI UX improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Security

If you find a security issue, please report it privately before opening a 
public issue. See [SECURITY.md](SECURITY.md).

---

## License

MIT
