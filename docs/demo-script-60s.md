# OpenShard 60-Second Demo Script

A tight script for recording or presenting a quick OpenShard demo. Uses the DocuVault production-infra demo folder so the pack context is clear to the viewer.

## Pre-recording checklist

- [ ] Clean terminal — no open TUI, no previous output visible
- [ ] Zoom level: large font, high contrast (viewer must read terminal output)
- [ ] Working directory: `examples/production-infra-demo` (not repo root — the IaC pack is strongest from here)
- [ ] No secrets visible in shell history, env, or `.openshard/` output — check before recording
- [ ] Use an existing recorded run for `openshard last --more` — avoid triggering another $0.30 model run

---

## Script

### Beat 1 — Hook (~12 seconds)

**Say:**
> "AI agents can write code and run commands. But do you know what changed, what it cost, or whether it was safe? OpenShard is the control layer for that."

**Viewer sees:** Empty terminal prompt.

---

### Beat 2 — Navigate and launch (~12 seconds)

**Say:**
> "This is the TUI. It ships with workflow packs — pre-built tasks for production-grade engineering work."

**Do:**
```bash
cd examples/production-infra-demo
openshard tui
```

In the TUI, type `/packs` and press Enter.

**Viewer sees:** TUI home screen, then the packs list (six packs rendered cleanly).

---

### Beat 3 — Pack detail (~12 seconds)

**Do in TUI:** `/pack production-iac-hardening`

**Say:**
> "Each pack is a controlled, reviewable task. OpenShard wraps the AI work with controls and receipts — you know what ran, what context was recorded, and what changed."

**Viewer sees:** Pack title, summary, tags, and recommended context.

---

### Beat 4 — Receipt (~12 seconds)

**Say:**
> "Here's the Shard receipt from the last run."

**Do:** Exit TUI. In shell:

```bash
openshard last --more
```

**Say:**
> "Model used. Context, when recorded. Changes. Findings, when structured findings exist. Cost. Feedback. Every run leaves one of these."

**Viewer sees:** Full SHARD block — Task, Execution, Context, Findings, Changes, Cost.

---

### Beat 5 — Close (~12 seconds)

**Say:**
> "OpenShard is the control layer for AI coding agents."

**Viewer sees:** Receipt still on screen.

---

## Notes

- Do not say "runs in a sandbox" for a read-only pack. Sandbox language applies only to write/apply tasks.
- Context and Findings may show "Not recorded" depending on run type — that is honest, not a bug.
- If `openshard last --more` is slow to render, cut to it pre-queued.
- If anything goes wrong mid-recording, `openshard last --more` still shows the receipt from the prior run.
