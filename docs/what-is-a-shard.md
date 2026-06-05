# What is a Shard?

**A Shard is the saved record of an AI coding run.**

When an AI tool writes or changes code, OpenShard records what happened: the task,
the model, the files touched, the checks run, the results, the risks, the approvals,
the cost, and a fingerprint of the saved record.

Think of it like a receipt for AI coding work.

## What a Shard proves

A Shard does not prove the code is perfect. Nothing can.

What it proves is more practical:

- what the AI was asked to do
- what OpenShard recorded during the run
- what changed
- what checks passed or failed
- whether anything risky was blocked
- whether approval was needed
- whether the saved record was changed later

That matters because AI coding should not be a black box.

## Checks

Checks are the tests and validations run during the work — things like formatting,
linting, or build steps. The Shard records, for each one, whether it **passed**,
**failed**, was **skipped**, or was **not run**. It does not hide a failed check.

## Trust score

The trust score is a single number that sums up how the run went at a glance — higher
when checks passed and nothing risky was blocked, lower when they didn't. It's a quick
signal to help you decide where to look first. It is a convenience, not a guarantee,
and it never replaces reading the run for yourself.

## The hash is a fingerprint

Every Shard gets a hash: a short fingerprint of the saved record. If the record is
changed later, the fingerprint no longer matches, so you can tell something moved.

That's all it is. The hash is **not a signature**, and it is **not blockchain proof**.
It's a simple way to check that the receipt you're reading is the one that was saved.

## Where OpenShard fits

OpenShard is not trying to beat Claude Code, Codex, Cursor, or OpenCode at writing
code. Those tools do the coding. OpenShard sits around the run and keeps the record of
what happened.

## The bigger picture

OpenShard starts with receipts. Over time it aims to grow into the control layer for
AI coding workflows — the place teams go to see, check, and stand behind the AI work
that touches their code.

Agents write code. OpenShard keeps the receipt.
