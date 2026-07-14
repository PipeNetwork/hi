from pathlib import Path

from solution import rotate_left

assert not Path(".git").exists()
values = [1, 2, 3, 4]
assert rotate_left(values, 1) == [2, 3, 4, 1]
assert rotate_left(values, 5) == [2, 3, 4, 1]
assert rotate_left(values, -1) == [4, 1, 2, 3]
assert rotate_left([], 99) == []
assert values == [1, 2, 3, 4]
