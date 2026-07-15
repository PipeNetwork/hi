import re

STRICT = re.compile(
    r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
)


def value(numeral):
    """Strict Roman numeral -> int (ValueError on malformed input)."""
    if not numeral or not STRICT.fullmatch(numeral):
        raise ValueError(f"malformed numeral: {numeral!r}")
    table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for a, b in zip(numeral, numeral[1:] + " "):
        v = table[a]
        total += -v if b != " " and v < table.get(b, 0) else v
    return total
