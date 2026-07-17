import re


def level(line):
    """Extract normalized log level per the task rules."""
    if line.lstrip().startswith("#"):
        return None
    m = re.search(
        r"\b(error|warn|warning|info|debug)\b", line, re.IGNORECASE
    )
    if not m:
        return "INFO"
    tok = m.group(1).upper()
    return "WARN" if tok == "WARNING" else tok
