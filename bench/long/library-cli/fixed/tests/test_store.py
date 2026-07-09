import os
import tempfile
import unittest
from datetime import date

import store
from catalog import Book, Catalog
from loans import Loans


class TestStore(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_load_missing_file_gives_empty(self):
        catalog, loans = store.load(self.path)
        self.assertEqual(catalog.search(), [])
        self.assertEqual(loans.overdue(date(2030, 1, 1)), [])

    def test_roundtrip(self):
        catalog = Catalog()
        catalog.add(Book("1", "Dune", "Frank Herbert", ["scifi"]))
        loans = Loans()
        loans.checkout("1", "ana", due_days=7, today=date(2026, 7, 1))
        store.save(catalog, loans, self.path)

        catalog2, loans2 = store.load(self.path)
        self.assertEqual(catalog2.get("1").author, "Frank Herbert")
        self.assertTrue(loans2.is_out("1"))
        self.assertEqual(loans2.overdue(date(2026, 7, 9)), ["1"])


if __name__ == "__main__":
    unittest.main()
