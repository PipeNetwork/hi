import re


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
