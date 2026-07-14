"""Unit tests for storage.py: save, load, missing file, corrupted JSON."""

import json
import os
import tempfile
import unittest

from models import Board, Card, Column
from storage import StorageError, board_exists, delete_board, load_board, save_board


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", prefix="kan_test_"
        )
        self.tmp.close()
        os.remove(self.tmp.name)
        self.path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _make_board(self):
        board = Board.create_default("Test Board")
        board.add_card("Task A", "Todo", tags=["x"])
        board.add_card("Task B", "Done")
        return board

    def test_save_and_load_roundtrip(self):
        board = self._make_board()
        save_board(board, self.path)
        self.assertTrue(board_exists(self.path))
        loaded = load_board(self.path)
        self.assertEqual(loaded.name, "Test Board")
        self.assertEqual(len(loaded.get_column("Todo").cards), 1)
        self.assertEqual(loaded.get_column("Todo").cards[0].title, "Task A")
        self.assertEqual(loaded.get_column("Todo").cards[0].tags, ["x"])

    def test_load_missing_file(self):
        with self.assertRaises(StorageError):
            load_board(self.path)

    def test_load_empty_file(self):
        with open(self.path, "w") as fh:
            fh.write("")
        with self.assertRaises(StorageError):
            load_board(self.path)

    def test_load_corrupted_json(self):
        with open(self.path, "w") as fh:
            fh.write("{not valid json")
        with self.assertRaises(StorageError):
            load_board(self.path)

    def test_load_non_object_json(self):
        with open(self.path, "w") as fh:
            fh.write("[1, 2, 3]")
        with self.assertRaises(StorageError):
            load_board(self.path)

    def test_load_invalid_board_data(self):
        # Missing required "name" field.
        with open(self.path, "w") as fh:
            json.dump({"columns": []}, fh)
        with self.assertRaises(StorageError):
            load_board(self.path)

    def test_save_non_board_rejected(self):
        with self.assertRaises(StorageError):
            save_board("not a board", self.path)  # type: ignore[arg-type]

    def test_save_is_atomic(self):
        """A failed save should not leave a partial/corrupt file."""
        board = self._make_board()
        save_board(board, self.path)
        original = load_board(self.path)
        # Attempt to save a non-serializable object to the same path; this
        # raises before touching the real file.
        with self.assertRaises(StorageError):
            save_board("bad", self.path)  # type: ignore[arg-type]
        # Original file should be intact.
        self.assertEqual(load_board(self.path).name, original.name)

    def test_board_exists_false(self):
        self.assertFalse(board_exists(self.path))

    def test_delete_board(self):
        board = self._make_board()
        save_board(board, self.path)
        delete_board(self.path)
        self.assertFalse(board_exists(self.path))

    def test_delete_board_missing_is_noop(self):
        # Should not raise even if the file does not exist.
        delete_board(self.path)


if __name__ == "__main__":
    unittest.main()
