from ledger.parser import parse
from ledger.account import Account
from ledger.report import summary

assert parse("2024-01-15 +12.50 groceries") == ("2024-01-15", 1250, "groceries")
assert parse("2024-01-16 -5.00 coffee") == ("2024-01-16", -500, "coffee")

a = Account()
a.apply("2024-01-15 +12.50 groceries")
a.apply("2024-01-16 -5.00 coffee")
assert a.balance() == 750, a.balance()
assert len(a.transactions()) == 2

rep = summary(a)
assert "Transactions: 2" in rep, rep
assert "Balance: $7.50" in rep, rep
print("ok")
