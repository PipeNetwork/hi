from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    sku: str
    quantity: int
    unit_cents: int


def parse_row(row: str) -> Record:
    sku, quantity, price = row.split("|")
    return Record(sku, int(quantity), int(float(price)))
