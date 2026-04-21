# Example 1: Simple refactoring task

## Task

```bash
openshard run "refactor the user validation function to be more readable" --write
```

## Routing decision

```
Task type: Low-risk refactoring
Complexity: Low
Security risk: None
Verification: Strong (existing tests cover this)

Routing: GLM-5.1 (cost-optimized)
Reason: Simple code cleanup, well-tested area, no security concerns
```

## Result

```
Done
Refactored user validation function with improved readability

Model: GLM-5.1 (single pass)

Files: 1 modified
  validators.py - Extracted helper functions, added descriptive names, reduced nesting

Time: 12.3s
Cost: $0.0043
```

## Comparison

| Approach | Model | Cost |
|----------|-------|------|
| OpenShard | GLM-5.1 | $0.004 |
| Always Sonnet | Sonnet 4.6 | $0.08 |
| Always Opus | Opus 4.6 | $0.15 |

**Savings: 95% vs Opus, 95% vs Sonnet**

## Why this worked

- Task was low-risk (refactoring existing, tested code)
- No security implications
- Strong verification (existing test suite would catch errors)
- GLM-5.1 handled it perfectly at 5% the cost