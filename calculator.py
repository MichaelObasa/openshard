def calculate_average(numbers):
    """
    Calculate the average of a list of numbers.
    """
    if not numbers:
        raise ValueError("Cannot calculate average of an empty list")

    total = sum(numbers)
    return total / len(numbers)


def factorial(n):
    """
    Calculate the factorial of a non-negative integer.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1

    result = 1
    for i in range(1, n + 1):
        result *= i

    return result
