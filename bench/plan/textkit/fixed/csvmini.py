def parse_csv(text):
    """Parse simple CSV into a list of rows (each a list of string fields).

    Rules: fields are comma-separated; a field may be double-quoted, in which
    case it may contain commas and newlines, and a doubled quote ("") inside a
    quoted field is a literal quote. Rows are newline-separated. A trailing
    newline does not produce an empty final row."""
    rows = []
    field = []
    row = []
    i = 0
    n = len(text)
    in_quotes = False
    saw_any = False
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            field.append(ch)
            i += 1
            continue
        if ch == '"':
            in_quotes = True
            saw_any = True
            i += 1
            continue
        if ch == ",":
            row.append("".join(field))
            field = []
            saw_any = True
            i += 1
            continue
        if ch == "\n":
            row.append("".join(field))
            rows.append(row)
            field = []
            row = []
            saw_any = False
            i += 1
            continue
        field.append(ch)
        saw_any = True
        i += 1
    if saw_any or field:
        row.append("".join(field))
        rows.append(row)
    return rows
