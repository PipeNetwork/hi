"""JSON persistence for the kan kanban board CLI.

Boards are persisted to a single ``kanban.json`` file. Provides load and save
operations with error handling for missing files and corrupted JSON.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from models import Board, ValidationError

DEFAULT_PATH = "kanban.json"


class StorageError(Exception):
    """Raised when persistence operations fail."""


def save_board(board: Board, path: str = DEFAULT_PATH) -> None:
    """Serialize ``board`` to ``path`` as JSON.

    Writes atomically by first writing a temporary file then renaming.
    """
    if not isinstance(board, Board):
        raise StorageError("save_board requires a Board instance")
    data = board.to_dict()
    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"failed to serialize board: {exc}") from exc
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        os.replace(tmp_path, path)
    except OSError as exc:
        # Clean up the temp file if the rename failed.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise StorageError(f"failed to write board to {path}: {exc}") from exc


def load_board(path: str = DEFAULT_PATH) -> Board:
    """Load and deserialize a Board from ``path``.

    Raises StorageError if the file is missing, empty, contains invalid JSON,
    or fails model validation.
    """
    if not os.path.exists(path):
        raise StorageError(f"board file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise StorageError(f"failed to read {path}: {exc}") from exc
    if not raw.strip():
        raise StorageError(f"board file is empty: {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageError(f"corrupted JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError(
            f"invalid board file: expected a JSON object, got {type(data).__name__}"
        )
    try:
        return Board.from_dict(data)
    except ValidationError as exc:
        raise StorageError(f"invalid board data in {path}: {exc}") from exc


def board_exists(path: str = DEFAULT_PATH) -> bool:
    """Return True if a board file exists at ``path``."""
    return os.path.exists(path)


def delete_board(path: str = DEFAULT_PATH) -> None:
    """Delete the board file at ``path`` if it exists."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise StorageError(f"failed to delete {path}: {exc}") from exc
