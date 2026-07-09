from metrics.tokenize import tokenize
from metrics.stats import word_count, unique_ratio


def report(text):
    toks = tokenize(text)
    return "words={} unique_ratio={:.2f}".format(word_count(toks), unique_ratio(toks))
