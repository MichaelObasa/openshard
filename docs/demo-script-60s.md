# OpenShard 60-Second Demo Script

A tight script for recording or presenting a quick OpenShard demo. Uses the OpenShard repo itself — no additional setup required.

## Pre-roll checklist

Before recording:

- [ ] OpenShard installed: `pip install openshard`
- [ ] In the openshard repo directory
- [ ] Terminal window visible, clean, no TUI open yet
- [ ] Font size large, contrast high

---

## Script

### Beat 1 — Hook (~12 seconds)

**Say:**
> "AI agents can write code and run commands. But do you know what changed? Was it safe? What did it cost? And is there a receipt you can audit later?"

**Viewer sees:** Empty terminal prompt.

---

### Beat 2 — Launch (~12 seconds)

**Say:**
> "This is OpenShard. Let me show you. I'll launch the TUI and give it a natural-language task."

**Do:**
```bash
openshard tui
```

Type in the TUI input: `explain this repo`

Press Enter.

**Viewer sees:** TUI launches. Task input visible. Task submitted.

---

### Beat 3 — Run (~12 seconds)

**Say:**
> "OpenShard runs this through its controlled workflow, routes it to the right model, and records everything — the task, the model, the steps, the cost."

**Viewer sees:** TUI shows the run in progress. Recent activity panel updates.

---

### Beat 4 — Receipt (~12 seconds)

**Say:**
> "Now I'll check the receipt."

**Do in TUI:** `/last more`

Or from the shell after exiting the TUI:

```bash
openshard last --more
```

**Say:**
> "Here it is: the task, the model that ran it, the cost, the checks, the outcome. Every run has one."

**Viewer sees:** Run receipt with task, model, cost, check results, status.

---

### Beat 5 — Close (~12 seconds)

**Say:**
> "Every AI action, tracked and auditable. You know what ran. You know what it cost. You can inspect it before anything touches your codebase. That's OpenShard."

**Viewer sees:** Receipt on screen.

---

## Notes

- Do not say "runs in a sandbox" for a read-only explain task. Use sandbox and dry-run language only for write and apply tasks.
- If the run completes quickly, pause briefly before showing the receipt — let the viewer absorb the TUI output.
- If anything goes wrong mid-recording, `openshard last --more` still shows the full receipt from a previous run.
