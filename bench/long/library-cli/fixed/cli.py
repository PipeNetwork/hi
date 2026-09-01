"""lib — a small library-management CLI."""

import argparse
import sys
from datetime import date

import store
from catalog import Book
from loans import LoanError


def build_parser():
    p = argparse.ArgumentParser(prog="lib", description="library catalog + loans")
    p.add_argument("--path", default=store.DEFAULT_PATH, help="data file")
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add a book")
    add.add_argument("isbn")
    add.add_argument("title")
    add.add_argument("author")
    add.add_argument("-t", "--tag", action="append", default=[], dest="tags")

    rm = sub.add_parser("remove", help="remove a book")
    rm.add_argument("isbn")

    search = sub.add_parser("search", help="search title/author")
    search.add_argument("query", nargs="?", default=None)
    search.add_argument("-t", "--tag", default=None)

    co = sub.add_parser("checkout", help="check a book out")
    co.add_argument("isbn")
    co.add_argument("member")
    co.add_argument("--days", type=int, default=14)

    ret = sub.add_parser("return", help="return a book")
    ret.add_argument("isbn")

    over = sub.add_parser("overdue", help="list overdue isbns")
    over.add_argument("--as-of", default=None, help="YYYY-MM-DD")

    sub.add_parser("list", help="list all books")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    catalog, loans = store.load(args.path)
    try:
        if args.command == "add":
            catalog.add(Book(args.isbn, args.title, args.author, args.tags))
            print(f"Added {args.title} ({args.isbn})")
        elif args.command == "remove":
            catalog.remove(args.isbn)
            print(f"Removed {args.isbn}")
        elif args.command == "search":
            for book in catalog.search(args.query, args.tag):
                status = "out" if loans.is_out(book.isbn) else "in"
                print(f"{book.isbn}  {book.title} — {book.author} [{status}]")
        elif args.command == "checkout":
            if catalog.get(args.isbn) is None:
                raise LoanError(f"unknown isbn: {args.isbn}")
            due = loans.checkout(args.isbn, args.member, args.days)
            print(f"Checked out {args.isbn} to {args.member}, due {due.isoformat()}")
        elif args.command == "return":
            loans.return_book(args.isbn)
            print(f"Returned {args.isbn}")
        elif args.command == "overdue":
            as_of = date.fromisoformat(args.as_of) if args.as_of else None
            for isbn in loans.overdue(as_of):
                print(isbn)
        elif args.command == "list":
            for book in catalog.search():
                print(f"{book.isbn}  {book.title} — {book.author}")
    except (ValueError, KeyError, LoanError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    store.save(catalog, loans, args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
