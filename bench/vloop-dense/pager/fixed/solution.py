def page(items, size, number):
    """1-based pagination per the task rules."""
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"bad size: {size!r}")
    if not isinstance(number, int) or number < 1:
        raise ValueError(f"bad page number: {number!r}")
    start = (number - 1) * size
    return list(items[start : start + size])
