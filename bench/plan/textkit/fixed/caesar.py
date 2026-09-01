def _shift(text, shift):
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") + shift) % 26 + ord("A")))
        else:
            out.append(ch)
    return "".join(out)


def encrypt(text, shift):
    """Caesar-shift letters forward by `shift`; non-letters unchanged."""
    return _shift(text, shift)


def decrypt(text, shift):
    """Inverse of encrypt for the same shift."""
    return _shift(text, -shift)
