from ledger.parser import parse


class Account:
    def __init__(self):
        self._txns = []

    def apply(self, line):
        self._txns.append(parse(line))

    def balance(self):
        return sum(cents for _, cents, _ in self._txns)

    def transactions(self):
        return list(self._txns)
