def clamp(x, lo, hi):
    """Constrain x to [lo, hi], swapping the bounds if given in reverse."""
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(x, hi))
