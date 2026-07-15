import re


def safe(name):
    """Sanitize a filename per the task rules."""
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "-", name)
    name = name.strip(".-")
    name = name[:24].rstrip(".-")
    return name or "untitled"
