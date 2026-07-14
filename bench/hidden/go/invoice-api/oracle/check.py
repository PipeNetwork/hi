import re
from pathlib import Path

model = Path("invoice/model.go").read_text()
total = Path("invoice/total.go").read_text()
assert re.search(r"\bUnitCents\s+int64\b", model)
assert re.search(r"\bQuantity\s+int64\b", model)
assert "UnitPrice" not in model + total
assert re.search(r"func\s+TotalCents\s*\(\s*items\s+\[\]LineItem\s*\)\s+int64", total)
assert "func Total(" not in total
assert re.search(r"(?:\.Quantity|quantity)\s*<\s*0", total)
assert re.search(r"UnitCents\s*\*\s*quantity", total, re.I)
