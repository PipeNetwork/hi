def rle_decode(s):
    """Decode a run-length-encoded string (char followed by a decimal count)."""
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        count = int(s[i:j])
        out.append(ch * count)
        i = j
    return "".join(out)
