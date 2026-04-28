---
name: API Rate Limiting
description: Patterns for adding rate limiting to HTTP APIs
category: performance
keywords: [rate-limit, throttle, api, endpoint, redis, backoff]
languages: [python, typescript]
---

Apply rate limits at the gateway or middleware layer, not inside business logic.
Use sliding-window counters in Redis for accurate per-user limits.
Return 429 with a Retry-After header so clients back off correctly.
Exempt health-check and metrics endpoints from rate limiting.
