import re
from pathlib import Path

routes = Path("src/routes.ts").read_text()
index = Path("src/index.ts").read_text()
assert re.search(r"export\s+const\s+API_PREFIX\s*=\s*[\"']/api/v2[\"']", routes)
assert re.search(r"export\s+const\s+projectRoute\s*=\s*`\$\{API_PREFIX}/projects`", routes)
assert re.search(r"userRoute\s*=\s*\(\s*id\s*:\s*string\s*\)", routes)
assert "encodeURIComponent(id)" in routes
assert "${API_PREFIX}/users/${encodeURIComponent(id)}" in routes
assert "/v1/" not in routes + index
for name in ("API_PREFIX", "projectRoute", "userRoute"):
    assert name in index
assert re.search(r"from\s*[\"']\./routes[\"']", index)
