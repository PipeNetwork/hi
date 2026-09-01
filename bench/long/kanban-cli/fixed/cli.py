"""Command-line interface for the kan kanban board CLI.

Subcommands: init, add, move, list, delete, stats.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from models import Board, Column, DEFAULT_COLUMNS, ValidationError
from storage import StorageError, board_exists, delete_board, load_board, save_board

DEFAULT_PATH = "kanban.json"


def _die(message: str, code: int = 1) -> None:
    """Print an error to stderr and exit."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _load_or_die(path: str) -> Board:
    try:
        return load_board(path)
    except StorageError as exc:
        _die(str(exc))


def _save_or_die(board: Board, path: str) -> None:
    try:
        save_board(board, path)
    except StorageError as exc:
        _die(str(exc))


def _parse_tags(tag_args: Optional[List[str]]) -> List[str]:
    """Parse --tag arguments. Each may be comma-separated."""
    if not tag_args:
        return []
    tags: List[str] = []
    for item in tag_args:
        for piece in item.split(","):
            piece = piece.strip()
            if piece:
                tags.append(piece)
    return tags


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    path = args.path
    if board_exists(path) and not args.force:
        _die(
            f"a board already exists at {path}; use --force to overwrite"
        )
    try:
        board = Board.create_default(name=args.name)
    except ValidationError as exc:
        _die(str(exc))
    _save_or_die(board, path)
    columns = ", ".join(DEFAULT_COLUMNS)
    print(f"Created board '{board.name}' at {path}")
    print(f"Columns: {columns}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    path = args.path
    board = _load_or_die(path)
    tags = _parse_tags(args.tag)
    try:
        card = board.add_card(
            title=args.title,
            column=args.column,
            description=args.description or "",
            tags=tags,
        )
    except ValidationError as exc:
        _die(str(exc))
    _save_or_die(board, path)
    tag_str = f" [{', '.join(card.tags)}]" if card.tags else ""
    print(f"Added card '{card.title}' (id: {card.id}) to '{card.column}'{tag_str}")
    return 0


def cmd_move(args: argparse.Namespace) -> int:
    path = args.path
    board = _load_or_die(path)
    try:
        card = board.move_card(args.card, args.to)
    except ValidationError as exc:
        _die(str(exc))
    _save_or_die(board, path)
    print(f"Moved card '{card.title}' (id: {card.id}) to '{card.column}'")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = args.path
    board = _load_or_die(path)
    tags = _parse_tags(args.tag)
    tag_set = {t.lower() for t in tags}

    print(f"Board: {board.name}")
    print("=" * 40)
    total = 0
    for col in board.columns:
        cards = col.cards
        if tag_set:
            cards = [c for c in cards if tag_set & {t.lower() for t in c.tags}]
        total += len(cards)
        print(f"\n{col.name} ({len(cards)})")
        print("-" * len(f"{col.name} ({len(cards)})"))
        if not cards:
            print("  (empty)")
        for card in cards:
            tag_str = f" [{', '.join(card.tags)}]" if card.tags else ""
            desc = f" — {card.description}" if card.description else ""
            print(f"  [{card.id}] {card.title}{tag_str}{desc}")
    print(f"\nTotal cards: {total}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    path = args.path
    board = _load_or_die(path)
    try:
        card = board.delete_card(args.card)
    except ValidationError as exc:
        _die(str(exc))
    _save_or_die(board, path)
    print(f"Deleted card '{card.title}' (id: {card.id}) from '{card.column}'")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    path = args.path
    board = _load_or_die(path)
    print(f"Board: {board.name}")
    print("=" * 40)
    print("\nCards per column:")
    total = 0
    for col in board.columns:
        count = len(col.cards)
        total += count
        bar = "#" * count
        print(f"  {col.name:<16} {count:>4}  {bar}")
    print(f"  {'TOTAL':<16} {total:>4}")

    # Tag distribution
    tag_counts = {}
    for card in board.all_cards():
        for tag in card.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\nTag distribution:")
    if not tag_counts:
        print("  (no tags)")
    else:
        for tag in sorted(tag_counts):
            print(f"  {tag:<16} {tag_counts[tag]:>4}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kan",
        description="A simple kanban board CLI.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"path to the board JSON file (default: {DEFAULT_PATH})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="create a new board with default columns")
    p_init.add_argument("--name", default="My Board", help="board name")
    p_init.add_argument(
        "--force", action="store_true", help="overwrite an existing board file"
    )
    p_init.set_defaults(func=cmd_init)

    # add
    p_add = sub.add_parser("add", help="add a card to a column")
    p_add.add_argument("title", help="card title")
    p_add.add_argument(
        "-c", "--column", default="Todo", help="target column (default: Todo)"
    )
    p_add.add_argument("-d", "--description", default="", help="card description")
    p_add.add_argument(
        "-t", "--tag", action="append", default=None,
        help="tag (may be repeated or comma-separated)",
    )
    p_add.set_defaults(func=cmd_add)

    # move
    p_move = sub.add_parser("move", help="move a card to another column")
    p_move.add_argument("card", help="card id or title")
    p_move.add_argument("to", help="target column name")
    p_move.set_defaults(func=cmd_move)

    # list
    p_list = sub.add_parser("list", help="list board contents")
    p_list.add_argument(
        "-t", "--tag", action="append", default=None,
        help="filter by tag (may be repeated or comma-separated)",
    )
    p_list.set_defaults(func=cmd_list)

    # delete
    p_delete = sub.add_parser("delete", help="delete a card")
    p_delete.add_argument("card", help="card id or title")
    p_delete.set_defaults(func=cmd_delete)

    # stats
    p_stats = sub.add_parser("stats", help="show board statistics")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        _die(f"unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
