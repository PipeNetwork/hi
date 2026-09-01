def render(inventory):
    return "\n".join(record.sku for record in inventory.records())
