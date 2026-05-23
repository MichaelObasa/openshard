from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _first_useful_line(stdout: str, stderr: str, fallback: str) -> str:
    for source in (stdout, stderr):
        for line in source.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    return fallback


def run_review_checks(path: Path) -> list[dict]:
    """Run safe, read-only checks for an IaC review.

    Returns a list of check result dicts. Never modifies files, never runs
    terraform init/apply/plan/destroy, never installs plugins, never hits the
    network. Returns [] when no Terraform files exist.
    """
    if not any(path.glob("**/*.tf")):
        return []

    results: list[dict] = []
    tf_bin = shutil.which("terraform")

    # Check 1: terraform fmt
    if tf_bin is None:
        results.append({
            "name": "terraform fmt",
            "status": "skipped",
            "command": "terraform fmt -check -recursive -no-color",
            "reason": "terraform not installed",
            "summary": "",
            "returncode": None,
        })
    else:
        try:
            proc = subprocess.run(
                [tf_bin, "fmt", "-check", "-recursive", "-no-color"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(path),
            )
            if proc.returncode == 0:
                results.append({
                    "name": "terraform fmt",
                    "status": "passed",
                    "command": "terraform fmt -check -recursive -no-color",
                    "reason": "",
                    "summary": "formatting is clean",
                    "returncode": 0,
                })
            else:
                summary = _first_useful_line(proc.stdout, proc.stderr, "formatting differences found")
                results.append({
                    "name": "terraform fmt",
                    "status": "failed",
                    "command": "terraform fmt -check -recursive -no-color",
                    "reason": "",
                    "summary": summary,
                    "returncode": proc.returncode,
                })
        except Exception:
            results.append({
                "name": "terraform fmt",
                "status": "skipped",
                "command": "terraform fmt -check -recursive -no-color",
                "reason": "check errored",
                "summary": "",
                "returncode": None,
            })

    # Check 2: terraform validate (only safe when .terraform/ already exists)
    if tf_bin is None:
        results.append({
            "name": "terraform validate",
            "status": "skipped",
            "command": "terraform validate -no-color",
            "reason": "terraform not installed",
            "summary": "",
            "returncode": None,
        })
    elif not (path / ".terraform").is_dir():
        results.append({
            "name": "terraform validate",
            "status": "skipped",
            "command": "terraform validate -no-color",
            "reason": "terraform init required",
            "summary": "",
            "returncode": None,
        })
    else:
        try:
            proc = subprocess.run(
                [tf_bin, "validate", "-no-color"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(path),
            )
            if proc.returncode == 0:
                summary = _first_useful_line(proc.stdout, proc.stderr, "configuration is valid")
                results.append({
                    "name": "terraform validate",
                    "status": "passed",
                    "command": "terraform validate -no-color",
                    "reason": "",
                    "summary": summary,
                    "returncode": 0,
                })
            else:
                summary = _first_useful_line(proc.stdout, proc.stderr, "validation failed")
                results.append({
                    "name": "terraform validate",
                    "status": "failed",
                    "command": "terraform validate -no-color",
                    "reason": "",
                    "summary": summary,
                    "returncode": proc.returncode,
                })
        except Exception:
            results.append({
                "name": "terraform validate",
                "status": "skipped",
                "command": "terraform validate -no-color",
                "reason": "check errored",
                "summary": "",
                "returncode": None,
            })

    # Check 3: tflint
    tflint_bin = shutil.which("tflint")
    if tflint_bin is None:
        results.append({
            "name": "tflint",
            "status": "skipped",
            "command": "tflint --no-color",
            "reason": "tflint not installed",
            "summary": "",
            "returncode": None,
        })
    else:
        try:
            proc = subprocess.run(
                [tflint_bin, "--no-color"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(path),
            )
            if proc.returncode == 0:
                results.append({
                    "name": "tflint",
                    "status": "passed",
                    "command": "tflint --no-color",
                    "reason": "",
                    "summary": "no issues found",
                    "returncode": 0,
                })
            else:
                summary = _first_useful_line(proc.stdout, proc.stderr, "issues found")
                results.append({
                    "name": "tflint",
                    "status": "failed",
                    "command": "tflint --no-color",
                    "reason": "",
                    "summary": summary,
                    "returncode": proc.returncode,
                })
        except Exception:
            results.append({
                "name": "tflint",
                "status": "skipped",
                "command": "tflint --no-color",
                "reason": "check errored",
                "summary": "",
                "returncode": None,
            })

    return results
