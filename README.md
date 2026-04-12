# Buggy Calculator — Demo Project

This project intentionally contains bugs to demonstrate how failing tests
guide you toward finding and fixing problems.

## Files

| File | Purpose |
|---|---|
| `calculator.py` | Two functions, each with a deliberate bug |
| `test_calculator.py` | pytest tests — most will **fail** until bugs are fixed |

## Bugs

### `calculate_average` — off-by-one in divisor

```python
# Buggy
return total / (len(numbers) + 1)

# Fixed
return total / len(numbers)
```

### `factorial` — loop excludes `n`

```python
# Buggy
for i in range(1, n):      # never multiplies by n

# Fixed
for i in range(1, n + 1):  # includes n
```

## Running the Tests

```bash
pip install pytest
pytest test_calculator.py -v
```

## Expected Test Results (before fixing)

```
FAILED test_calculator.py::TestCalculateAverage::test_average_of_simple_list
FAILED test_calculator.py::TestCalculateAverage::test_average_of_single_element
FAILED test_calculator.py::TestCalculateAverage::test_average_of_identical_elements
FAILED test_calculator.py::TestCalculateAverage::test_average_of_negative_numbers
PASSED test_calculator.py::TestCalculateAverage::test_average_raises_on_empty_list

PASSED test_calculator.py::TestFactorial::test_factorial_of_zero
PASSED test_calculator.py::TestFactorial::test_factorial_of_one   <- accidentally passes
FAILED test_calculator.py::TestFactorial::test_factorial_of_five
FAILED test_calculator.py::TestFactorial::test_factorial_of_six
FAILED test_calculator.py::TestFactorial::test_factorial_of_ten
PASSED test_calculator.py::TestFactorial::test_factorial_raises_on_negative
```

7 failed, 4 passed.
