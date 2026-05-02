import sys
from csv_utils import parse_csv_row

doc = parse_csv_row.__doc__
if not doc or len(doc.strip()) < 10:
    print("parse_csv_row is missing a meaningful docstring")
    sys.exit(1)

print("OK")
