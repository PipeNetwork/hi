from collections import Counter


def most_common(xs):
    """Most frequent element of xs; ties broken by smallest element."""
    counts = Counter(xs)
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)
