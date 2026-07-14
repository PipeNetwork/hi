from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Record:
    sku: str
    quantity: int
    unit_cents: int


def parse_row(row: str) -> Record:
    sku, quantity, price = (part.strip() for part in row.split("|"))
    cents = int(Decimal(price) * 100)
    return Record(sku, int(quantity), cents)
