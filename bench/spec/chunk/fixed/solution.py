def chunk(items, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    return [items[i : i + n] for i in range(0, len(items), n)]
