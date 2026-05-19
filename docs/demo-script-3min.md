# OpenShard 3-Minute Demo Script

A deeper demo showing OpenShard on production-shaped engineering work. Uses the DocuVault fictional IaC hardening scenario from `examples/production-infra-demo/`.

## Pre-roll checklist

Before recording:

- [ ] OpenShard installed: `pip install openshard`
- [ ] In the openshard repo root (so the path below resolves)
- [ ] Terminal window visible, clean, no TUI open yet
- [ ] `examples/production-infra-demo/demo-task.md` open and ready to copy
- [ ] Font size large, contrast high

---

## Act 1 — The TUI (30 seconds)

**Say:**
> "Let me show you OpenShard. This is the TUI — the control layer that sits in front of your AI agent."

**Do:**
```bash
cd examples/production-infra-demo
openshard tui
```

**Say:**
> "You've got a task input, run controls, and a recent activity panel that tracks everything that's run. OpenShard isn't the AI — it's what wraps the AI and makes sure you know what happened."

**Viewer sees:** TUI home screen. Task input at top, recent activity panel, run controls visible.

---

## Act 2 — Running a production-shaped task (45 seconds)

**Say:**
> "I'm going to give it a real engineering task. Not a toy example. This is a hardening review of a deliberately flawed Terraform codebase — a fictional document-processing platform called DocuVault."

**Do:** Paste the task from `examples/production-infra-demo/demo-task.md` into the TUI input:

```
Review and harden this deliberately flawed Terraform codebase.
Assess it through three lenses: security and compliance posture
for a bank-facing product, operability for a 2am incident with
no infrastructure specialist on call, and developer experience
for a 5–10 person engineering team. Identify critical, high,
and medium risks. Explain trade-offs. Propose safe hardening
changes. Prioritise ruthlessly. Do not apply without review.
```

Press Enter.

**Say:**
> "OpenShard routes this to the right model and runs the task through its controlled workflow. Three lenses: security for a bank-facing product, 2am operability, and developer experience for a small team."

**Viewer sees:** Task submitted. TUI shows run in progress. Model visible in activity.

---

## Act 3 — Recent activity (30 seconds)

**Say:**
> "While it runs — you can see the recent activity panel. Every previous run is here: timestamped, costed, with its outcome. This is your audit trail."

**Do in TUI:** `/last`

**Say:**
> "Here's the previous run. Task, model, duration, cost, and whether it passed checks."

**Viewer sees:** Recent activity panel. `/last` shows the last run receipt inline.

---

## Act 4 — The receipt (30 seconds)

**Do:** Exit TUI. Run in shell:

```bash
openshard last --more
```

**Say:**
> "This is the full receipt. Task, model, tokens, cost, checks, verification outcome, files proposed. Every AI-assisted action goes through OpenShard and leaves a record you can audit — not just output you hope was right."

**Viewer sees:** Full run receipt in terminal. Model, cost, check results, proposed files listed.

---

## Act 5 — Diff and safety (30 seconds)

**Say:**
> "Before anything touches your repo, inspect the diff."

**Do:**
```bash
openshard diff-last
```

**Say:**
> "Here are the exact changes it proposed. Now I'll run a dry-run apply — previews what would happen with no changes to disk."

**Do:**
```bash
openshard apply-last --dry-run
```

**Say:**
> "Dry run complete. It tells you exactly what would be written if you applied it. You decide whether to proceed."

**Viewer sees:** Unified diff. Then dry-run output listing files that would be written.

---

## Close (15 seconds)

**Say:**
> "Terraform changes, IAM policies, deployment scripts — AI can help with all of it. OpenShard makes sure you know what happened, whether it was safe, and what it cost. Before it touches production. That's the receipt. That's OpenShard."

**Viewer sees:** Terminal with last command output.

---

## Notes

- Use `examples/production-infra-demo/` as the demo repo, or a sanitised equivalent.
- Do not include real project IDs, real IPs, real secrets, or real company names.
- Pause after each command — let the viewer read the output before continuing.
- If the task runs longer than expected, continue talking through Act 3 while it finishes.
- The DocuVault Terraform is deliberately flawed — OpenShard will have real findings to show.
