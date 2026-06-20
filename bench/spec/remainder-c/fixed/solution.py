def remainder_c(a, b):
    """Remainder of a divided by b, with the sign of the dividend (C-style)."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return a - b * q
