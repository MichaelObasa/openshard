def parse_csv_row(line: str) -> list:
    return [field.strip() for field in line.split(",")]
