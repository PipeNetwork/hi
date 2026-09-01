_VALUES = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def to_roman(n):
    """Integer 1..3999 -> Roman numeral string."""
    if not isinstance(n, int) or not (1 <= n <= 3999):
        raise ValueError("n must be an integer in 1..3999")
    out = []
    for value, symbol in _VALUES:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def from_roman(s):
    """Roman numeral string -> integer."""
    symbols = {sym: val for val, sym in [(1, "I"), (5, "V"), (10, "X"), (50, "L"), (100, "C"), (500, "D"), (1000, "M")]}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        if ch not in symbols:
            raise ValueError(f"bad roman numeral: {s!r}")
        val = symbols[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total
