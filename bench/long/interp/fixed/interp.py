"""A small infix expression interpreter: tokenizer + recursive-descent parser."""


def _tokenize(s):
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            text = s[i:j]
            toks.append(("num", float(text) if "." in text else int(text)))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            toks.append(("name", s[i:j]))
            i = j
            continue
        two = s[i : i + 2]
        if two in ("**", "<=", ">=", "==", "!="):
            toks.append(("op", two))
            i += 2
            continue
        if c in "+-*/%<>(),":
            toks.append(("op", c))
            i += 1
            continue
        raise ValueError("bad character: " + repr(c))
    toks.append(("end", None))
    return toks


_FUNCS = {"min": min, "max": max, "abs": lambda x: abs(x)}


class _Parser:
    def __init__(self, toks, env):
        self.toks = toks
        self.pos = 0
        self.env = env

    def _peek(self):
        return self.toks[self.pos]

    def _advance(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def _match(self, *ops):
        t = self._peek()
        if t[0] == "op" and t[1] in ops:
            self.pos += 1
            return t[1]
        return None

    def _expect(self, op):
        if self._match(op) is None:
            raise ValueError("expected " + op)

    def parse(self):
        v = self.comparison()
        if self._peek()[0] != "end":
            raise ValueError("trailing tokens")
        return v

    def comparison(self):
        left = self.additive()
        op = self._match("<", ">", "<=", ">=", "==", "!=")
        if op is None:
            return left
        right = self.additive()
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        return left != right

    def additive(self):
        left = self.multiplicative()
        while True:
            op = self._match("+", "-")
            if op is None:
                return left
            right = self.multiplicative()
            left = left + right if op == "+" else left - right

    def multiplicative(self):
        left = self.power()
        while True:
            op = self._match("*", "/", "%")
            if op is None:
                return left
            right = self.power()
            if op == "*":
                left = left * right
            elif op == "/":
                left = left / right
            else:
                left = left % right

    def power(self):
        left = self.unary()
        if self._match("**") is not None:
            right = self.power()  # right-associative
            return left**right
        return left

    def unary(self):
        op = self._match("+", "-")
        if op == "-":
            return -self.unary()
        if op == "+":
            return self.unary()
        return self.primary()

    def primary(self):
        t = self._peek()
        if t[0] == "num":
            self._advance()
            return t[1]
        if self._match("(") is not None:
            v = self.comparison()
            self._expect(")")
            return v
        if t[0] == "name":
            self._advance()
            name = t[1]
            if self._match("(") is not None:
                args = []
                if self._peek() != ("op", ")"):
                    args.append(self.comparison())
                    while self._match(",") is not None:
                        args.append(self.comparison())
                self._expect(")")
                if name not in _FUNCS:
                    raise NameError("unknown function: " + name)
                return _FUNCS[name](*args)
            if name in self.env:
                return self.env[name]
            raise NameError("undefined variable: " + name)
        raise ValueError("unexpected token: " + repr(t))


def evaluate(expr, env=None):
    return _Parser(_tokenize(expr), env or {}).parse()
