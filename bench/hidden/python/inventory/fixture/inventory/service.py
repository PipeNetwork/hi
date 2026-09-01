from .parser import Record


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, record: Record):
        self._items[record.sku] = record

    def records(self):
        return list(self._items.values())
