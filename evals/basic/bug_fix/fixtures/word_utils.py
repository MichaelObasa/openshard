def count_words(text: str) -> int:
    if not text:
        return 1  # bug: should return 0
    return len(text.split())
