import unittest

from catalog import Book, Catalog


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.cat = Catalog()
        self.cat.add(Book("1", "Dune", "Frank Herbert", ["scifi"]))
        self.cat.add(Book("2", "Emma", "Jane Austen", ["classic"]))

    def test_add_and_get(self):
        self.assertEqual(self.cat.get("1").title, "Dune")

    def test_duplicate_isbn_rejected(self):
        with self.assertRaises(ValueError):
            self.cat.add(Book("1", "Other", "Someone"))

    def test_required_fields(self):
        with self.assertRaises(ValueError):
            Book("", "t", "a")

    def test_remove(self):
        self.cat.remove("2")
        self.assertIsNone(self.cat.get("2"))
        with self.assertRaises(KeyError):
            self.cat.remove("2")

    def test_search_title_case_insensitive(self):
        self.assertEqual([b.isbn for b in self.cat.search("dune")], ["1"])

    def test_search_author(self):
        self.assertEqual([b.isbn for b in self.cat.search("austen")], ["2"])

    def test_search_tag_filter(self):
        self.assertEqual([b.isbn for b in self.cat.search(tag="scifi")], ["1"])
        self.assertEqual(self.cat.search("dune", tag="classic"), [])

    def test_search_all_sorted_by_title(self):
        self.assertEqual([b.title for b in self.cat.search()], ["Dune", "Emma"])

    def test_roundtrip(self):
        again = Catalog.from_dict(self.cat.to_dict())
        self.assertEqual(sorted(again.books), ["1", "2"])
        self.assertEqual(again.get("1").tags, ["scifi"])


if __name__ == "__main__":
    unittest.main()
