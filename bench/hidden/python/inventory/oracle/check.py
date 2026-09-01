from inventory import Inventory, parse_row, render

record = parse_row(" A-1 | 2 | 10.25 ")
assert (record.sku, record.quantity, record.unit_cents) == ("A-1", 2, 1025)
inventory = Inventory()
inventory.add(record)
inventory.add(parse_row("B-2|1|2.50"))
inventory.add(parse_row("A-1|3|10.25"))
assert [(r.sku, r.quantity, r.unit_cents) for r in inventory.records()] == [
    ("A-1", 5, 1025),
    ("B-2", 1, 250),
]
assert render(inventory) == (
    "A-1 qty=5 value=$51.25\n"
    "B-2 qty=1 value=$2.50\n"
    "TOTAL $53.75"
)
try:
    inventory.add(parse_row("A-1|1|9.99"))
except ValueError:
    pass
else:
    raise AssertionError("conflicting price was accepted")
