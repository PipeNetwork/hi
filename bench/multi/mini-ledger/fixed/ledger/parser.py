def parse(line):
    """Parse 'DATE ±DOLLARS DESC' -> (date, cents, desc)."""
    date, amount, desc = line.split(maxsplit=2)
    return date, round(float(amount) * 100), desc
