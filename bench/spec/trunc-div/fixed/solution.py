def divide(a, b):
    """Integer-divide a by b, truncating toward zero."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q
