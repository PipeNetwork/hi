#!/usr/bin/env python3
"""Hidden scorer for the `textkit` /goal plan-completion fixture.

Each of the six plan sections is one module with an exact contract. A section
counts as *delivered* only if its module imports and every check passes. The
score is `delivered / 6` — a fractional "% of the plan actually built", so a run
that finishes 4 of 6 sections scores 0.67 instead of a flat fail.

Runs against the candidate's modules in the current working directory. Prints a
per-section PASS/FAIL breakdown and a machine-readable `HI_EVAL_SCORE=<0..1>`
line, and exits 0 only when all six sections are delivered.
"""
import importlib
import os
import sys
import traceback

# Import the candidate's modules from the workspace, not this scorer's dir.
sys.path.insert(0, os.getcwd())


def check_slugify():
    m = importlib.import_module("slugify")
    assert m.slugify("Hello, World!") == "hello-world"
    assert m.slugify("  A_B  c ") == "a-b-c"
    assert m.slugify("--Rock & Roll--") == "rock-roll"
    assert m.slugify("") == ""


def check_wordcount():
    m = importlib.import_module("wordcount")
    assert m.word_count("The the THE cat") == {"the": 3, "cat": 1}
    assert m.word_count("Hi, hi! Bye.") == {"hi": 2, "bye": 1}
    assert m.word_count("") == {}


def check_roman():
    m = importlib.import_module("roman")
    assert m.to_roman(4) == "IV"
    assert m.to_roman(1994) == "MCMXCIV"
    assert m.to_roman(2023) == "MMXXIII"
    assert m.from_roman("MCMXCIV") == 1994
    assert m.from_roman("XLII") == 42
    for n in (1, 9, 40, 90, 400, 900, 3999, 1888):
        assert m.from_roman(m.to_roman(n)) == n
    for bad in (0, 4000, -1):
        try:
            m.to_roman(bad)
            raise AssertionError(f"to_roman({bad}) should raise ValueError")
        except ValueError:
            pass


def check_caesar():
    m = importlib.import_module("caesar")
    assert m.encrypt("abc XYZ!", 3) == "def ABC!"
    for text, k in (("Hello, World!", 5), ("zZ aA", 1), ("nothing here 123", 13)):
        assert m.decrypt(m.encrypt(text, k), k) == text


def check_rpn():
    m = importlib.import_module("rpn")
    assert m.eval_rpn("3 4 + 2 *") == 14.0
    assert m.eval_rpn("10 2 /") == 5.0
    assert m.eval_rpn("5 1 2 + 4 * + 3 -") == 14.0
    for bad in ("1 +", "1 2 3", "1 x +"):
        try:
            m.eval_rpn(bad)
            raise AssertionError(f"eval_rpn({bad!r}) should raise ValueError")
        except ValueError:
            pass


def check_csvmini():
    m = importlib.import_module("csvmini")
    assert m.parse_csv("a,b,c") == [["a", "b", "c"]]
    assert m.parse_csv('a,b\n"c,d",e') == [["a", "b"], ["c,d", "e"]]
    assert m.parse_csv('x,"he said ""hi"""') == [["x", 'he said "hi"']]
    assert m.parse_csv("a,b\n") == [["a", "b"]]


SECTIONS = [
    ("slugify", check_slugify),
    ("wordcount", check_wordcount),
    ("roman", check_roman),
    ("caesar", check_caesar),
    ("rpn", check_rpn),
    ("csvmini", check_csvmini),
]


def main():
    delivered = 0
    print("=== textkit plan-completion score ===")
    for name, check in SECTIONS:
        try:
            check()
            delivered += 1
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 — a broken/missing module must not crash scoring
            first = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"[FAIL] {name}: {first}")
    total = len(SECTIONS)
    pct = round(100 * delivered / total)
    print(f"sections delivered: {delivered}/{total} ({pct}%)")
    print(f"HI_EVAL_SCORE={delivered / total:.4f}")
    sys.exit(0 if delivered == total else 1)


if __name__ == "__main__":
    main()
