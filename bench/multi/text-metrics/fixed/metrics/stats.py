def word_count(tokens):
    return len(tokens)


def unique_ratio(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0.0
