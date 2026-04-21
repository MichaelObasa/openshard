# Example 3: Complex multi-stage task

## Task

```bash
openshard run "build an async task queue API with worker processing, job status endpoints, and full test coverage" --write --verify
```

## Routing decision

```
Task type: Multi-stage system implementation
Complexity: High (architecture + async patterns + distributed concerns)
Security risk: Medium (API auth, job isolation)
Verification: Strong (comprehensive test coverage requested)

Multi-model routing:
  Architecture & planning → Sonnet 4.6 (strong reasoning)
  Async worker implementation → Kimi K2.5 (long-context, agent swarm strength)
  API endpoints (boilerplate) → DeepSeek V3.2 (cost-efficient generation)
  Test generation → DeepSeek V3.2 (structured, repetitive)
  Review & integration → Sonnet 4.6 (verification)
```

## Execution stages

### Stage 1: Architecture & planning
```
Model: Sonnet 4.6
Time: 14.2s
Cost: $0.0184

Output:
  System design document
  Database schema
  API contract
  Task queue architecture
```

### Stage 2: Async worker implementation
```
Model: Kimi K2.5
Time: 82.4s
Cost: $0.1156

Why Kimi: Long-running implementation with multiple cooperating components.
Kimi's agent swarm approach handles distributed worker coordination better
than sequential generation.

Output:
  worker.py - Async job processor with retry logic
  queue.py - Redis-backed task queue with priority support
  task_handlers.py - Job execution framework
```

### Stage 3: API endpoints (boilerplate)
```
Model: DeepSeek V3.2
Time: 31.6s
Cost: $0.0421

Why DeepSeek: Standard CRUD endpoints, well-defined patterns, low risk.
No need for expensive model here.

Output:
  api/jobs.py - Create, status, cancel endpoints
  api/health.py - Health check and metrics
  middleware/auth.py - API key validation
```

### Stage 4: Test generation
```
Model: DeepSeek V3.2
Time: 18.7s
Cost: $0.0298

Output:
  tests/test_worker.py - Worker execution tests
  tests/test_queue.py - Queue operations tests
  tests/test_api.py - API endpoint tests
  tests/integration/ - Full flow integration tests
```

### Stage 5: Review & integration
```
Model: Sonnet 4.6
Time: 9.3s
Cost: $0.0127

Verification: All tests passing (47/47)
```

## Final result

```
Done
Built async task queue API with worker processing, job status endpoints,
and full test coverage

Models used:
  Architecture: Sonnet 4.6
  Worker implementation: Kimi K2.5
  API endpoints: DeepSeek V3.2
  Tests: DeepSeek V3.2
  Review: Sonnet 4.6

Files: 12 created
Total time: 156.2s
Total cost: $0.3086
```

## Comparison

| Approach | Models | Cost | Why the difference |
|----------|--------|------| -------------------|
| OpenShard | Multi-model routing | $0.31 | Right model for each stage |
| Always Sonnet | Sonnet 4.6 | $0.68 | Overpays for boilerplate |
| Always Opus | Opus 4.6 | $1.20 | Massive overpay |
| Always cheap | DeepSeek V3.2 | $0.18 | Would fail on architecture/workers |

**Savings: 74% vs always-Opus, 54% vs always-Sonnet**

## Why multi-model routing matters

This task had distinct stages with different requirements:

- **Architecture** needed strong reasoning → Sonnet 4.6
- **Async workers** needed coordinated generation → Kimi K2.5
- **API boilerplate** was low-risk, repetitive → DeepSeek V3.2
- **Tests** were structured generation → DeepSeek V3.2
- **Review** needed verification strength → Sonnet 4.6

Using one model for everything would either waste money (Opus) or produce lower-quality architecture/workers (DeepSeek). OpenShard routed intelligently and saved 54-74% while maintaining quality.