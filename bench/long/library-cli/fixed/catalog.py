"""Book catalog: add/remove/lookup/search."""


class Book:
    def __init__(self, isbn, title, author, tags=None):
        if not isbn or not title or not author:
            raise ValueError("isbn, title, and author are required")
        self.isbn = isbn
        self.title = title
        self.author = author
        self.tags = list(tags or [])

    def to_dict(self):
        return {
            "isbn": self.isbn,
            "title": self.title,
            "author": self.author,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["isbn"], d["title"], d["author"], d.get("tags", []))


class Catalog:
    def __init__(self):
        self.books = {}

    def add(self, book):
        if book.isbn in self.books:
            raise ValueError(f"duplicate isbn: {book.isbn}")
        self.books[book.isbn] = book

    def remove(self, isbn):
        if isbn not in self.books:
            raise KeyError(isbn)
        del self.books[isbn]

    def get(self, isbn):
        return self.books.get(isbn)

    def search(self, query=None, tag=None):
        """Case-insensitive substring match on title/author; optional tag filter."""
        out = []
        q = (query or "").lower()
        for book in self.books.values():
            if q and q not in book.title.lower() and q not in book.author.lower():
                continue
            if tag and tag not in book.tags:
                continue
            out.append(book)
        return sorted(out, key=lambda b: b.title.lower())

    def to_dict(self):
        return {isbn: b.to_dict() for isbn, b in self.books.items()}

    @classmethod
    def from_dict(cls, d):
        cat = cls()
        for entry in d.values():
            cat.add(Book.from_dict(entry))
        return cat
