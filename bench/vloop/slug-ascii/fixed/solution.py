import re


def slug(title):
    """ASCII slug: lowercase, hyphen-separated; non-ASCII drops out."""
    ascii_only = "".join(c for c in title if ord(c) < 128)
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
