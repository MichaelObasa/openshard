from word_utils import count_words


def test_empty_string():
    assert count_words("") == 0


def test_single_word():
    assert count_words("hello") == 1


def test_multiple_words():
    assert count_words("one two three") == 3
