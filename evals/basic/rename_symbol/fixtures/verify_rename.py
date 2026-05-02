import importlib
import sys
from pathlib import Path

try:
    import math_utils

    importlib.reload(math_utils)
    fn = getattr(math_utils, "calculate_total", None)
    if fn is None:
        print("calculate_total not found in math_utils")
        sys.exit(1)
    assert fn(3, 4) == 7, "calculate_total(3, 4) should return 7"
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

if hasattr(math_utils, "calc"):
    print("calc is still defined in math_utils — rename incomplete")
    sys.exit(1)

main_src = Path("main.py").read_text()
if "calc(" in main_src:
    print("main.py still contains calc( — update incomplete")
    sys.exit(1)

print("OK")
