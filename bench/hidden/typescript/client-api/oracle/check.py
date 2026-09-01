import re
from pathlib import Path

client = Path("src/client.ts").read_text()
index = Path("src/index.ts").read_text()
assert re.search(r"export\s+interface\s+User\s*{", client)
assert re.search(
    r"export\s+async\s+function\s+loadUser\s*\(\s*id\s*:\s*string\s*\)\s*:\s*Promise\s*<\s*User\s*>",
    client,
)
assert "encodeURIComponent(id)" in client
assert re.search(r"/users/\$\{\s*encodeURIComponent\(id\)\s*}", client)
assert "fetchUser" not in client + index
assert re.search(r"export\s*{\s*loadUser\s*}\s*from\s*[\"']\./client[\"']", index)
assert re.search(r"export\s+type\s*{\s*User\s*}\s*from\s*[\"']\./client[\"']", index)
