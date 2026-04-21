# OpenShard Examples

Real tasks run with OpenShard, showing different routing decisions.

## Quick reference

| Example | Task type | Models used | Time | Cost | Key insight |
|---------|-----------|-------------|------|------|-------------|
| [Simple refactor](simple-refactor.md) | Low-risk code cleanup | GLM-5.1 | 12s | $0.004 | Cheap model handles simple work |
| [Security-sensitive auth](security-sensitive-auth.md) | JWT implementation with retry | Sonnet 4.6 → Opus 4.6 (escalation) | 94s | $0.18 | Auto-escalates on failure |
| [Complex multi-stage](complex-multi-stage.md) | API with async workers + tests | Sonnet 4.6 + Kimi K2.5 + DeepSeek V3.2 | 156s | $0.31 | Different models per stage |

## What these show

- **Simple refactor**: OpenShard uses the cheapest capable model when risk is low
- **Security-sensitive auth**: Starts with balanced model, escalates to strongest when needed
- **Complex multi-stage**: Routes different stages to models that excel at that specific work

Each example includes the full output, routing decisions, and cost comparison.