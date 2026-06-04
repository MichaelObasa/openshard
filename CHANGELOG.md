# Changelog

All notable changes to OpenShard are documented here.

## Unreleased

### Added

- Shard schema hardening with backward-compatible coercion and blocked unsafe raw fields
- Evidence provenance records for Shard proof claims
- Timeline and policy provenance in machine-readable proof output
- Machine-readable proof output for latest run inspection
- CI policy check mode with pass/warn/fail/skip results and deterministic exit behavior
- GitHub Actions-compatible output surfaces for CI receipt surfacing
- Run trust score with scored band and per-penalty breakdown
- Shard completeness stats with field-level scoring and band thresholds
- Failure taxonomy stats with category classification across run history
- Failed Shard to eval case generation for local eval authoring
- Repo map with bounded metadata walk and schema versioning
- Repo map cache stored at `.openshard/cache/` for repeated planning runs
- Repo-aware plan mode using local repo context metadata
- Machine-readable run timeline export
- Hardened developer interaction trace with sanitized field output
- First-run onboarding commands for initial setup
- Shared sanitization helpers for safe Shard and provenance field export
- `NOTICE` file for Apache-2.0 attribution

### Changed

- License changed from MIT to Apache-2.0
- Runtime gate decision ordering unified across execution path

### Fixed

- CI check GitHub Actions output test isolation
- Repo map path sanitisation on Linux

## 0.1.2 - 2026-06-01

- Published OpenShard 0.1.2 to PyPI
- Confirmed clean `pipx install openshard` path
- Fixed package config defaults for clean out-of-the-box install
- Improved package/source command parity
- Included proof-flow commands through the installed package
