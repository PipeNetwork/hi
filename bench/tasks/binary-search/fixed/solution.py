def binary_search(xs, target):
    """Return the index of target in sorted list xs, or -1 if absent."""
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1
