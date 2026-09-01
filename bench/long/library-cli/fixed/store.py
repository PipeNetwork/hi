"""JSON persistence for the catalog + loans."""

import json
import os

from catalog import Catalog
from loans import Loans

DEFAULT_PATH = "library.json"


def save(catalog, loans, path=DEFAULT_PATH):
    data = {"catalog": catalog.to_dict(), "loans": loans.to_dict()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load(path=DEFAULT_PATH):
    if not os.path.exists(path):
        return Catalog(), Loans()
    with open(path) as f:
        data = json.load(f)
    return Catalog.from_dict(data.get("catalog", {})), Loans.from_dict(data.get("loans", {}))
