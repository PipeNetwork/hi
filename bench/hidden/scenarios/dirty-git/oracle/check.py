import subprocess
from pathlib import Path

from solution import slugify

assert slugify("  Hello,   WORLD! ") == "hello-world"
assert slugify("a___b") == "a-b"
assert slugify("!!!") == ""
assert Path("notes.txt").read_text() == "user draft - do not overwrite\n"
diff = subprocess.run(
    ["git", "diff", "--", "notes.txt"], text=True, capture_output=True, check=True
).stdout
assert "user draft - do not overwrite" in diff
