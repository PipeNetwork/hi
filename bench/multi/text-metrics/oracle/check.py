from metrics.tokenize import tokenize
from metrics.stats import word_count, unique_ratio
from metrics.report import report

assert tokenize("Hello, HELLO world!") == ["hello", "hello", "world"]
assert word_count(["a", "b", "c"]) == 3
assert abs(unique_ratio(["a", "a", "b", "b"]) - 0.5) < 1e-9
assert report("Hello hello world") == "words=3 unique_ratio=0.67", report("Hello hello world")
print("ok")
