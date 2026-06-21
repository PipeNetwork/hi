def parse_bool(s):
    t = s.strip().lower()
    if t in ("true", "yes", "1", "on"):
        return True
    if t in ("false", "no", "0", "off"):
        return False
    raise ValueError(f"not a boolean: {s!r}")
