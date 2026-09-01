_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def eval_rpn(expr):
    """Evaluate a space-separated reverse-Polish expression and return a float.
    Supports + - * / on numbers. Raises ValueError on a malformed expression."""
    stack = []
    for token in expr.split():
        if token in _OPS:
            if len(stack) < 2:
                raise ValueError(f"not enough operands for {token!r}")
            b = stack.pop()
            a = stack.pop()
            stack.append(_OPS[token](a, b))
        else:
            try:
                stack.append(float(token))
            except ValueError:
                raise ValueError(f"bad token: {token!r}")
    if len(stack) != 1:
        raise ValueError("malformed expression")
    return stack[0]
