# DocuVault — Flagship Demo Task

Paste the text below into the OpenShard TUI input field and press Enter.

---

Review and harden this deliberately flawed Terraform codebase.

Assess it through three lenses:
1. Security and compliance posture for a bank-facing product
2. Operability for a 2am incident with no infrastructure specialist on call
3. Developer experience for a 5–10 person engineering team that is not infrastructure-specialist

Identify critical, high, and medium risks. Explain trade-offs. Propose safe hardening changes.
Prioritise ruthlessly. Do not apply changes directly without review.

---

After OpenShard completes the run, inspect the receipt:

```
openshard last --more
```

To preview any proposed changes:

```
openshard diff-last
```

To dry-run apply:

```
openshard apply-last --dry-run
```
