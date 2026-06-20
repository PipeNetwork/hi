def titlecase(s):
    """Capitalize the first letter of each space-separated word, lowercase rest.

    Only the first letter of each whitespace-delimited word changes, so
    hyphenated words like 'well-known' become 'Well-known' (unlike str.title()).
    """
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))
