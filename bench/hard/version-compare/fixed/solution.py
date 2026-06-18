def compare_versions(a, b):
    """Compare dotted version strings; return -1, 0, or 1."""
    pa = [int(x) for x in a.split(".")]
    pb = [int(x) for x in b.split(".")]
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    for x, y in zip(pa, pb):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0
