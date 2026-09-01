def flatten(xs):
    """Flatten one level of nesting: [[1, 2], [3]] -> [1, 2, 3]."""
    result = []
    for sub in xs:
        result.extend(sub)
    return result
