def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and vals[s[i + 1]] > vals[ch]:
            total -= vals[ch]
        else:
            total += vals[ch]
    return total
