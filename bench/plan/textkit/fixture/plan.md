# textkit — a small Python text-utilities library

Build **textkit**, a Python library of six independent, self-contained modules in
this directory. Each section below is one module with an exact behavioral
contract and examples. Implement every section fully; a section counts as
delivered only when its module imports and its functions behave exactly as
specified. Pure standard library only (no third-party packages). Do not create a
package directory — put each module as a top-level `.py` file in this directory.

## 1. `slugify.py`

`slugify(s: str) -> str`: lowercase the string, replace every maximal run of
characters that are not `a`–`z` or `0`–`9` with a single hyphen `-`, then strip
leading and trailing hyphens.

- `slugify("Hello, World!")` → `"hello-world"`
- `slugify("  A_B  c ")` → `"a-b-c"`
- `slugify("--Rock & Roll--")` → `"rock-roll"`

## 2. `wordcount.py`

`word_count(s: str) -> dict`: return a dict mapping each word to how many times
it occurs. A word is a maximal run of `a`–`z` or `0`–`9`, compared
case-insensitively (lowercase the keys).

- `word_count("The the THE cat")` → `{"the": 3, "cat": 1}`
- `word_count("Hi, hi! Bye.")` → `{"hi": 2, "bye": 1}`
- `word_count("")` → `{}`

## 3. `roman.py`

Two functions for classic Roman numerals (values 1..3999):

- `to_roman(n: int) -> str`: integer → numeral. `to_roman(4)` → `"IV"`,
  `to_roman(1994)` → `"MCMXCIV"`, `to_roman(2023)` → `"MMXXIII"`. Raise
  `ValueError` if `n` is not an integer in `1..3999`.
- `from_roman(s: str) -> int`: numeral → integer (accept upper case).
  `from_roman("MCMXCIV")` → `1994`, `from_roman("XLII")` → `42`.

`from_roman(to_roman(n)) == n` for every `n` in `1..3999`.

## 4. `caesar.py`

A Caesar cipher over ASCII letters:

- `encrypt(text: str, shift: int) -> str`: shift each letter forward by `shift`
  (wrapping within `a`–`z` and within `A`–`Z`); leave every non-letter
  unchanged. `encrypt("abc XYZ!", 3)` → `"def ABC!"`.
- `decrypt(text: str, shift: int) -> str`: the inverse, so
  `decrypt(encrypt(t, k), k) == t` for any text `t` and shift `k`.

## 5. `rpn.py`

`eval_rpn(expr: str) -> float`: evaluate a space-separated reverse-Polish
expression supporting `+ - * /` over numbers, returning a float. Division is
floating-point. Raise `ValueError` on a malformed expression (bad token, too few
operands, or leftover operands).

- `eval_rpn("3 4 + 2 *")` → `14.0`
- `eval_rpn("10 2 /")` → `5.0`
- `eval_rpn("5 1 2 + 4 * + 3 -")` → `14.0`

## 6. `csvmini.py`

`parse_csv(text: str) -> list`: parse simple CSV into a list of rows, each a list
of string fields. Fields are comma-separated; a field may be double-quoted, in
which case it may contain commas and newlines; a doubled quote `""` inside a
quoted field is a literal `"`. Rows are newline-separated, and a trailing newline
does not create an empty final row.

- `parse_csv("a,b,c")` → `[["a", "b", "c"]]`
- `parse_csv('a,b\n"c,d",e')` → `[["a", "b"], ["c,d", "e"]]`
- `parse_csv('x,"he said ""hi"""')` → `[["x", 'he said "hi"']]`
