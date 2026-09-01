def int_to_roman(n):
    """Convert a positive integer (1..3999) to a Roman numeral string."""
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for value, symbol in vals:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
