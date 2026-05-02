from string_utils import truncate_string


def test_short_string_unchanged():
    assert truncate_string("hi", 10) == "hi"


def test_long_string_truncated():
    assert truncate_string("hello world", 5) == "hello..."


def test_custom_suffix():
    assert truncate_string("hello world", 5, suffix="!") == "hello!"


def test_exact_length_unchanged():
    assert truncate_string("hello", 5) == "hello"
