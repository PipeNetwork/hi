def render(fields):
    """Render one CSV line per the quoting rules in the task."""
    out = []
    for f in fields:
        if f is None:
            out.append("")
            continue
        f = str(f)
        needs = (
            "," in f
            or '"' in f
            or "\n" in f
            or f != f.strip(" ")
        )
        if needs:
            f = '"' + f.replace('"', '""') + '"'
        out.append(f)
    return ",".join(out)
