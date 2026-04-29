from pathlib import Path

content = Path("README.md").read_text(encoding="utf-8")
assert "--dry-run" in content, "README.md is missing --dry-run documentation"
print("PASS: README.md documents --dry-run")
