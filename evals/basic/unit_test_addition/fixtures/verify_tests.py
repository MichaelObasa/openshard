import ast
import subprocess
import sys
from pathlib import Path

source = Path("tests/test_palindrome.py").read_text()
tree = ast.parse(source)
test_fns = [
    n.name
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
]
if len(test_fns) < 4:
    print(f"Expected at least 4 test_ functions, found {len(test_fns)}: {test_fns}")
    sys.exit(1)

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_palindrome.py", "-q"]
)
sys.exit(result.returncode)
