import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "bug"))
import solution as s

assert s.add(2, 3) == 5
assert s.add(0, 0) == 0
print("ok")
