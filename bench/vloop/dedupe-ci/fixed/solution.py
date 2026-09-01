def dedupe(words):
    """Case-insensitive dedupe preserving order and first-seen spelling."""
    seen, out = set(), []
    for word in words:
        key = word.casefold()
        if key not in seen:
            seen.add(key)
            out.append(word)
    return out
