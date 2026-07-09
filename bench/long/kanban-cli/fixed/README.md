# kan

A simple kanban board command-line tool written in pure Python. Boards are
persisted to a single JSON file (`kanban.json` by default) — no database, no
server, no dependencies beyond the standard library.

## Features

- **Boards** with ordered columns (defaults: `Todo`, `In Progress`, `Done`)
- **Cards** with a title, description, and tags
- **Move** cards between columns by id or title
- **List** the board, optionally filtered by tag
- **Stats** showing counts per column and a tag distribution
- **Validation** for names, tags, and state transitions (a card can't move
  back out of `Done` into `Todo`/`In Progress`)
- Atomic JSON persistence with error handling for missing/corrupt files

## Requirements

- Python 3.8+

No third-party packages are required.

## Installation

`kan` is a single-package CLI. Clone or copy the three source files
(`models.py`, `storage.py`, `cli.py`) into a directory and invoke it as a
module:

```sh
python3 cli.py --help
```

For convenience, add a shell alias:

```sh
alias kan='python3 /path/to/cli.py'
```

## Usage

All commands accept a global `--path` option to point at a different board
file (default: `kanban.json` in the current directory).

### `init` — create a new board

```sh
python3 cli.py init --name "My Project"
```

Creates `kanban.json` with the default columns. Use `--force` to overwrite an
existing board:

```sh
python3 cli.py init --name "Fresh Start" --force
```

### `add` — add a card

```sh
python3 cli.py add "Write the README" -c "Todo" -d "Document every command" -t docs -t writing
```

Options:

| Flag | Description |
|------|-------------|
| `-c, --column` | Target column (default: `Todo`) |
| `-d, --description` | Card description |
| `-t, --tag` | A tag; may be repeated or comma-separated (`-t a,b`) |

Tags are normalized to lowercase, deduplicated, and sorted. Valid tag
characters are letters, digits, hyphens, and underscores.

### `move` — move a card

```sh
python3 cli.py move "Write the README" "In Progress"
# or by id:
python3 cli.py move 3f1a9c2e "Done"
```

A card in `Done` cannot be moved back to `Todo` or `In Progress` (this is a
guarded state transition). Moving a card to its current column is a no-op.

### `list` — show the board

```sh
python3 cli.py list
```

Filter by tag (matches any of the given tags):

```sh
python3 cli.py list -t docs
python3 cli.py list -t docs,writing
python3 cli.py list -t docs -t urgent
```

### `delete` — remove a card

```sh
python3 cli.py delete "Write the README"
# or by id:
python3 cli.py delete 3f1a9c2e
```

### `stats` — board statistics

```sh
python3 cli.py stats
```

Prints a per-column card count (with a simple bar chart) and a tag
distribution.

## Sample workflow

```sh
# Start a new board
python3 cli.py init --name "Website Launch"

# Plan some work
python3 cli.py add "Design homepage" -t design -t frontend
python3 cli.py add "Set up CI" -t devops -d "GitHub Actions pipeline"
python3 cli.py add "Write blog post" -t content

# Begin work
python3 cli.py move "Design homepage" "In Progress"

# See where things stand
python3 cli.py list
python3 cli.py stats

# Filter to just the frontend work
python3 cli.py list -t frontend

# Finish and clean up
python3 cli.py move "Design homepage" "Done"
python3 cli.py delete "Set up CI"
```

## File format

The board is stored as JSON with this shape:

```json
{
  "name": "My Project",
  "columns": [
    {
      "name": "Todo",
      "cards": [
        {
          "id": "3f1a9c2e",
          "title": "Write the README",
          "description": "Document every command",
          "tags": ["docs", "writing"],
          "column": "Todo"
        }
      ]
    },
    { "name": "In Progress", "cards": [] },
    { "name": "Done", "cards": [] }
  ]
}
```

## Testing

Run the full test suite with:

```sh
python3 -m unittest discover -s tests -t .
```

Tests cover model validation, state transitions, storage (save, load, missing
file, corrupted JSON, atomic writes), and every CLI subcommand including tag
filtering edge cases.
