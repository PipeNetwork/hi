import re


def word_count(s):
    """Return a dict mapping each word to its count. Words are maximal runs of
    [a-z0-9], compared case-insensitively."""
    counts = {}
    for word in re.findall(r"[a-z0-9]+", s.lower()):
        counts[word] = counts.get(word, 0) + 1
    return counts
