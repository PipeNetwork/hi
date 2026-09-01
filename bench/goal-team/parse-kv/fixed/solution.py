def parse_kv(s):
    out = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"malformed entry: {part!r}")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out
