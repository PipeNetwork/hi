def count_vowels(s):
    """Count the vowels (a, e, i, o, u) in s, case-insensitively."""
    vowels = "aeio"
    return sum(1 for c in s if c in vowels)
