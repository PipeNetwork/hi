import re
from pathlib import Path

source = Path("model/user.go").read_text()
body = re.search(r"type\s+User\s+struct\s*{(?P<body>.*?)}", source, re.S)
assert body, "User struct missing"
fields = body.group("body")
expected = {
    "ID": ("int", 'json:"id"'),
    "DisplayName": ("string", 'json:"display_name"'),
    "Email": ("string", 'json:"email,omitempty"'),
}
for name, (kind, tag) in expected.items():
    assert re.search(rf"\b{name}\s+{kind}\s+`{re.escape(tag)}`", fields), name
assert fields.count("omitempty") == 1
assert "user_id" not in fields and "displayName" not in fields
