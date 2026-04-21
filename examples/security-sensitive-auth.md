# Example 2: Security-sensitive authentication (with retry)

## Task

```bash
openshard run "add JWT authentication with login endpoint and token helpers" --write --verify
```

## Routing decision

```
Task type: Security-sensitive implementation
Complexity: Medium-high
Security risk: High (auth, tokens, session management)
Verification: Medium (tests can catch logic errors, not all security issues)

Initial routing: Sonnet 4.6 (balanced strength + cost)
Reason: Security-sensitive work needs reliable model, but not necessarily the most expensive
```

## Initial attempt

```
Planning (Sonnet 4.6): 7.8s, $0.0037
Implementation (Sonnet 4.6): 78.7s, $0.1144

Files: 6 created
Verification: FAILED
Error: Token signature validation missing edge case handling
```

## Retry with escalation

```
Retrying with stronger model...
Routing: Opus 4.6 (escalated for reliability)

Fix attempt (Opus 4.6): 18.4s, $0.0521

Verification: PASSED
All tests passing (18/18)
```

## Final result

```
Done
Added JWT authentication with login endpoint, token helpers, middleware,
and protected routes

Models used:
  Planning: Sonnet 4.6
  Implementation: Sonnet 4.6 (initial)
  Fix: Opus 4.6 (escalated)

Files: 6 created
  jwt_helpers.py - Token generation with proper validation
  auth.py - Login endpoint and session handling
  app.py - Protected routes with auth middleware
  tests/test_jwt_helpers.py - Token edge cases
  tests/test_auth.py - Login flow integration tests
  requirements.txt - Flask, PyJWT, pytest

Total time: 94.2s
Total cost: $0.1702
```

## Comparison

| Approach | Models | Cost | Success on first try? |
|----------|--------|------|-----------------------|
| OpenShard | Sonnet → Opus (retry) | $0.17 | Yes (after 1 retry) |
| Always Sonnet | Sonnet 4.6 | $0.18 | Maybe (would need manual retry) |
| Always Opus | Opus 4.6 | $0.42 | Likely |

**Savings: 60% vs always-Opus**

## Why retry/escalation matters

OpenShard didn't waste money using Opus for planning and initial implementation. When verification caught an issue, it automatically escalated to the stronger model for the fix. You got the reliability of Opus where it mattered, at a fraction of the cost.