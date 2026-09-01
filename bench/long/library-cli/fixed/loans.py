"""Loan tracking: checkout/return/overdue, one active loan per book."""

from datetime import date, timedelta


class LoanError(Exception):
    pass


class Loans:
    def __init__(self):
        # isbn -> {"member": str, "due": "YYYY-MM-DD"}
        self.active = {}

    def checkout(self, isbn, member, due_days=14, today=None):
        if isbn in self.active:
            raise LoanError(f"{isbn} is already checked out")
        if due_days <= 0:
            raise ValueError("due_days must be positive")
        start = today or date.today()
        due = start + timedelta(days=due_days)
        self.active[isbn] = {"member": member, "due": due.isoformat()}
        return due

    def return_book(self, isbn):
        if isbn not in self.active:
            raise LoanError(f"{isbn} is not checked out")
        del self.active[isbn]

    def is_out(self, isbn):
        return isbn in self.active

    def member_loans(self, member):
        return sorted(i for i, l in self.active.items() if l["member"] == member)

    def overdue(self, as_of=None):
        """isbns whose due date is strictly before `as_of` (default today)."""
        cutoff = (as_of or date.today()).isoformat()
        return sorted(i for i, l in self.active.items() if l["due"] < cutoff)

    def to_dict(self):
        return dict(self.active)

    @classmethod
    def from_dict(cls, d):
        loans = cls()
        loans.active = dict(d)
        return loans
