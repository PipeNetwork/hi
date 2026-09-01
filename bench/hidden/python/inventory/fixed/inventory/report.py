def render(inventory):
    records = sorted(inventory.records(), key=lambda record: record.sku)
    lines = [
        f"{record.sku} qty={record.quantity} value=${record.quantity * record.unit_cents / 100:.2f}"
        for record in records
    ]
    total = sum(record.quantity * record.unit_cents for record in records)
    lines.append(f"TOTAL ${total / 100:.2f}")
    return "\n".join(lines)
