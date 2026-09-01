def summary(account):
    return "Transactions: {}\nBalance: ${:.2f}".format(
        len(account.transactions()), account.balance() / 100
    )
