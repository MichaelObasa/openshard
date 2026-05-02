from config_utils import parse_config


def test_basic_key_value():
    assert parse_config("HOST=localhost") == {"HOST": "localhost"}


def test_comment_lines_ignored():
    result = parse_config("# this is a comment\nPORT=8080")
    assert result == {"PORT": "8080"}


def test_blank_lines_ignored():
    result = parse_config("\nKEY=val\n\n")
    assert result == {"KEY": "val"}


def test_multiple_keys():
    result = parse_config("A=1\nB=2")
    assert result == {"A": "1", "B": "2"}
