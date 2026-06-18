def count_vowels(s):
    """Count the vowels (a, e, i, o, u) in s, case-insensitively."""
    vowels = "aeiou"
    return sum(1 for c in s.lower() if c in vowels)
