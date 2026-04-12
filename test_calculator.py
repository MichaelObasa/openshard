import pytest
from calculator import calculate_average, factorial


# ---------------------------------------------------------------------------
# Tests for calculate_average
# ---------------------------------------------------------------------------

class TestCalculateAverage:

    def test_average_of_simple_list(self):
        """Average of [1, 2, 3] should be 2.0, but the bug returns 1.5."""
        result = calculate_average([1, 2, 3])
        assert result == 2.0, f"Expected 2.0 but got {result}"

    def test_average_of_single_element(self):
        """Average of [10] should be 10.0, but the bug returns 5.0."""
        result = calculate_average([10])
        assert result == 10.0, f"Expected 10.0 but got {result}"

    def test_average_of_identical_elements(self):
        """Average of [4, 4, 4, 4] should be 4.0, but the bug returns 3.2."""
        result = calculate_average([4, 4, 4, 4])
        assert result == 4.0, f"Expected 4.0 but got {result}"

    def test_average_of_negative_numbers(self):
        """Average of [-3, -1, -2] should be -2.0, but the bug returns -1.5."""
        result = calculate_average([-3, -1, -2])
        assert result == -2.0, f"Expected -2.0 but got {result}"

    def test_average_raises_on_empty_list(self):
        """Should raise ValueError for an empty list — this test should pass."""
        with pytest.raises(ValueError, match="Cannot calculate average of an empty list"):
            calculate_average([])


# ---------------------------------------------------------------------------
# Tests for factorial
# ---------------------------------------------------------------------------

class TestFactorial:

    def test_factorial_of_zero(self):
        """0! should be 1 — this test should pass (special case)."""
        assert factorial(0) == 1

    def test_factorial_of_one(self):
        """1! should be 1, but the bug returns 1 (range(1,1) is empty — accidentally correct)."""
        # NOTE: This accidentally passes because range(1, 1) is empty and result stays 1.
        assert factorial(1) == 1

    def test_factorial_of_five(self):
        """5! should be 120, but the bug computes 4! = 24 instead."""
        result = factorial(5)
        assert result == 120, f"Expected 120 but got {result}"

    def test_factorial_of_six(self):
        """6! should be 720, but the bug computes 5! = 120 instead."""
        result = factorial(6)
        assert result == 720, f"Expected 720 but got {result}"

    def test_factorial_of_ten(self):
        """10! should be 3628800, but the bug computes 9! = 362880 instead."""
        result = factorial(10)
        assert result == 3628800, f"Expected 3628800 but got {result}"

    def test_factorial_raises_on_negative(self):
        """Should raise ValueError for negative input — this test should pass."""
        with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
            factorial(-1)
