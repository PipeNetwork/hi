import re


def slugify(s):
    """Lowercase; collapse each run of non-[a-z0-9] characters into one
    hyphen; strip leading/trailing hyphens."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")
