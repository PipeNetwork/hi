import re


def parse(text):
    """Duration string like '2h45m30s' -> total seconds (int)."""
    if not text or not re.fullmatch(r"(\d+[dhms])+", text):
        raise ValueError(f"bad duration: {text!r}")
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    seen, total = set(), 0
    for num, unit in re.findall(r"(\d+)([dhms])", text):
        if unit in seen:
            raise ValueError(f"repeated unit {unit!r}")
        seen.add(unit)
        total += int(num) * mult[unit]
    return total
