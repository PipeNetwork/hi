from .parser import Record


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, record: Record):
        previous = self._items.get(record.sku)
        if previous is not None:
            if previous.unit_cents != record.unit_cents:
                raise ValueError("conflicting unit price")
            record = Record(record.sku, previous.quantity + record.quantity, record.unit_cents)
        self._items[record.sku] = record

    def records(self):
        return list(self._items.values())
