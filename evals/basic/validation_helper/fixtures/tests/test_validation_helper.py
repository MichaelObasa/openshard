import pytest
from age_utils import parse_age


def test_valid_age():
    assert parse_age(25) == 25


def test_zero_is_valid():
    assert parse_age(0) == 0


def test_max_valid_age():
    assert parse_age(150) == 150


def test_negative_raises():
    with pytest.raises(ValueError):
        parse_age(-1)


def test_over_max_raises():
    with pytest.raises(ValueError):
        parse_age(151)
