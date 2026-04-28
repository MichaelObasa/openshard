---
name: Auth Hardening
description: Guidance for implementing and reviewing authentication code safely
category: security
keywords: [auth, login, jwt, session, token, password, oauth]
languages: [python, typescript, javascript]
framework: flask
---

Always hash passwords with bcrypt or argon2 — never store plaintext or MD5.
Validate JWTs on every protected route; check expiry, issuer, and signature.
Rotate refresh tokens on use and invalidate old ones server-side.
Set secure, httpOnly, sameSite=strict on session cookies.
Rate-limit login endpoints and log failed attempts with IP and timestamp.
