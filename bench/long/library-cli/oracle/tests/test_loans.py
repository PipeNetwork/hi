import unittest
from datetime import date

from loans import LoanError, Loans


class TestLoans(unittest.TestCase):
    def setUp(self):
        self.loans = Loans()
        self.today = date(2026, 7, 1)

    def test_checkout_sets_due_date(self):
        due = self.loans.checkout("1", "ana", due_days=7, today=self.today)
        self.assertEqual(due, date(2026, 7, 8))
        self.assertTrue(self.loans.is_out("1"))

    def test_double_checkout_rejected(self):
        self.loans.checkout("1", "ana", today=self.today)
        with self.assertRaises(LoanError):
            self.loans.checkout("1", "bob", today=self.today)

    def test_due_days_positive(self):
        with self.assertRaises(ValueError):
            self.loans.checkout("1", "ana", due_days=0, today=self.today)

    def test_return(self):
        self.loans.checkout("1", "ana", today=self.today)
        self.loans.return_book("1")
        self.assertFalse(self.loans.is_out("1"))
        with self.assertRaises(LoanError):
            self.loans.return_book("1")

    def test_member_loans(self):
        self.loans.checkout("2", "ana", today=self.today)
        self.loans.checkout("1", "ana", today=self.today)
        self.loans.checkout("3", "bob", today=self.today)
        self.assertEqual(self.loans.member_loans("ana"), ["1", "2"])

    def test_overdue_strictly_before_cutoff(self):
        self.loans.checkout("1", "ana", due_days=7, today=self.today)   # due 07-08
        self.loans.checkout("2", "bob", due_days=30, today=self.today)  # due 07-31
        self.assertEqual(self.loans.overdue(date(2026, 7, 8)), [])
        self.assertEqual(self.loans.overdue(date(2026, 7, 9)), ["1"])

    def test_roundtrip(self):
        self.loans.checkout("1", "ana", today=self.today)
        again = Loans.from_dict(self.loans.to_dict())
        self.assertTrue(again.is_out("1"))
        self.assertEqual(again.member_loans("ana"), ["1"])


if __name__ == "__main__":
    unittest.main()
