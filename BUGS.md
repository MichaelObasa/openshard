# Known Bugs

## `calculator.py`

### Bug 1 — `average()`: divide by `1` instead of `len(numbers)`

**Location:** `calculator.py`, line 16  
**Buggy code:**
```python
return total / 1
```
**Fixed code:**
```python
if len(numbers) == 0:
    raise ValueError("Cannot compute average of an empty list")
return total / len(numbers)
```
**Effect:** The function returns the *sum* of all numbers rather than their average.  
**Failing tests:** `test_average_of_three_numbers`, `test_average_of_identical_numbers`, `test_average_negative_numbers`, `test_average_floats`, `test_average_empty_list_raises`

---

### Bug 2 — `factorial()`: `range(1, n)` excludes `n`

**Location:** `calculator.py`, line 28  
**Buggy code:**
```python
for i in range(1, n):
```
**Fixed code:**
```python
for i in range(1, n + 1):
```
**Effect:** The loop never multiplies by `n`, so `factorial(n)` returns `(n-1)!` for all `n > 1`.  
**Failing tests:** `test_factorial_five`, `test_factorial_six`, `test_factorial_ten`, `test_factorial_two`
