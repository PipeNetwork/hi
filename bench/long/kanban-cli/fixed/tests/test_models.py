"""Unit tests for models.py: validation rules and state transitions."""

import unittest

from models import (
    Board,
    Card,
    Column,
    DEFAULT_COLUMNS,
    ValidationError,
    validate_name,
    validate_tag,
    validate_tags,
    validate_title,
)


class TestNameValidation(unittest.TestCase):
    def test_valid_name_is_stripped(self):
        self.assertEqual(validate_name("  Todo  "), "Todo")

    def test_empty_name_rejected(self):
        with self.assertRaises(ValidationError):
            validate_name("   ")

    def test_non_string_name_rejected(self):
        with self.assertRaises(ValidationError):
            validate_name(123)  # type: ignore[arg-type]

    def test_overlong_name_rejected(self):
        with self.assertRaises(ValidationError):
            validate_name("x" * 101)


class TestTitleValidation(unittest.TestCase):
    def test_valid_title_stripped(self):
        self.assertEqual(validate_title("  My Task  "), "My Task")

    def test_empty_title_rejected(self):
        with self.assertRaises(ValidationError):
            validate_title("")

    def test_overlong_title_rejected(self):
        with self.assertRaises(ValidationError):
            validate_title("x" * 201)


class TestTagValidation(unittest.TestCase):
    def test_valid_tag_lowercased(self):
        self.assertEqual(validate_tag("  Python  "), "python")

    def test_hyphen_underscore_allowed(self):
        self.assertEqual(validate_tag("high-priority"), "high-priority")
        self.assertEqual(validate_tag("needs_review"), "needs_review")

    def test_empty_tag_rejected(self):
        with self.assertRaises(ValidationError):
            validate_tag("")

    def test_invalid_chars_rejected(self):
        with self.assertRaises(ValidationError):
            validate_tag("bad tag!")
        with self.assertRaises(ValidationError):
            validate_tag("bad.tag")

    def test_tags_deduplicated_and_sorted(self):
        self.assertEqual(validate_tags(["python", "Python", "api", "api"]),
                         ["api", "python"])

    def test_too_many_tags_rejected(self):
        with self.assertRaises(ValidationError):
            validate_tags([f"t{i}" for i in range(11)])

    def test_none_tags_returns_empty(self):
        self.assertEqual(validate_tags(None), [])


class TestCard(unittest.TestCase):
    def test_card_creation(self):
        card = Card(title="Task", description="desc", tags=["py", "api"])
        self.assertEqual(card.title, "Task")
        self.assertEqual(card.description, "desc")
        self.assertEqual(card.tags, ["api", "py"])
        self.assertTrue(card.id)

    def test_card_roundtrip(self):
        card = Card(title="Task", tags=["py"])
        data = card.to_dict()
        restored = Card.from_dict(data)
        self.assertEqual(restored.title, card.title)
        self.assertEqual(restored.id, card.id)
        self.assertEqual(restored.tags, card.tags)

    def test_card_from_dict_missing_title(self):
        with self.assertRaises(ValidationError):
            Card.from_dict({"id": "abc"})

    def test_has_tag_case_insensitive(self):
        card = Card(title="T", tags=["python"])
        self.assertTrue(card.has_tag("Python"))
        self.assertFalse(card.has_tag("ruby"))


class TestColumn(unittest.TestCase):
    def test_column_add_and_remove(self):
        col = Column(name="Todo")
        card = Card(title="T1")
        col.add_card(card)
        self.assertEqual(len(col.cards), 1)
        self.assertEqual(card.column, "Todo")
        removed = col.remove_card(card.id)
        self.assertIsNotNone(removed)
        self.assertEqual(len(col.cards), 0)

    def test_remove_nonexistent_returns_none(self):
        col = Column(name="Todo")
        self.assertIsNone(col.remove_card("nope"))

    def test_find_card_by_id_and_title(self):
        col = Column(name="Todo")
        card = Card(title="My Task")
        col.add_card(card)
        self.assertIs(col.find_card(card.id), card)
        self.assertIs(col.find_card("my task"), card)
        self.assertIsNone(col.find_card("other"))

    def test_column_roundtrip(self):
        col = Column(name="Todo")
        col.add_card(Card(title="A"))
        col.add_card(Card(title="B", tags=["x"]))
        restored = Column.from_dict(col.to_dict())
        self.assertEqual(restored.name, "Todo")
        self.assertEqual(len(restored.cards), 2)
        self.assertEqual(restored.cards[1].tags, ["x"])


class TestBoard(unittest.TestCase):
    def test_create_default(self):
        board = Board.create_default("Work")
        self.assertEqual(board.name, "Work")
        self.assertEqual(board.column_names(), DEFAULT_COLUMNS)

    def test_duplicate_column_rejected(self):
        with self.assertRaises(ValidationError):
            Board(name="B", columns=[Column("Todo"), Column("todo")])

    def test_add_card(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo", tags=["a"])
        self.assertEqual(card.column, "Todo")
        self.assertIn(card, board.get_column("Todo").cards)

    def test_add_card_unknown_column(self):
        board = Board.create_default()
        with self.assertRaises(ValidationError):
            board.add_card("Task", "Nonexistent")

    def test_move_card(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        moved = board.move_card(card.id, "In Progress")
        self.assertEqual(moved.column, "In Progress")
        self.assertEqual(len(board.get_column("Todo").cards), 0)
        self.assertEqual(len(board.get_column("In Progress").cards), 1)

    def test_move_card_by_title(self):
        board = Board.create_default()
        board.add_card("Task", "Todo")
        moved = board.move_card("task", "Done")
        self.assertEqual(moved.column, "Done")

    def test_move_card_unknown_target(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        with self.assertRaises(ValidationError):
            board.move_card(card.id, "Nowhere")

    def test_move_card_not_found(self):
        board = Board.create_default()
        with self.assertRaises(ValidationError):
            board.move_card("missing", "Done")

    def test_move_done_back_to_todo_forbidden(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        board.move_card(card.id, "Done")
        with self.assertRaises(ValidationError):
            board.move_card(card.id, "Todo")
        with self.assertRaises(ValidationError):
            board.move_card(card.id, "In Progress")

    def test_move_within_same_column_allowed(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        moved = board.move_card(card.id, "Todo")
        self.assertEqual(moved.column, "Todo")

    def test_move_done_to_done_allowed(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        board.move_card(card.id, "Done")
        moved = board.move_card(card.id, "Done")
        self.assertEqual(moved.column, "Done")

    def test_delete_card(self):
        board = Board.create_default()
        card = board.add_card("Task", "Todo")
        deleted = board.delete_card(card.id)
        self.assertEqual(deleted.id, card.id)
        self.assertEqual(len(board.get_column("Todo").cards), 0)

    def test_delete_card_not_found(self):
        board = Board.create_default()
        with self.assertRaises(ValidationError):
            board.delete_card("missing")

    def test_board_roundtrip(self):
        board = Board.create_default("Proj")
        board.add_card("A", "Todo", tags=["x"])
        board.add_card("B", "In Progress")
        restored = Board.from_dict(board.to_dict())
        self.assertEqual(restored.name, "Proj")
        self.assertEqual(restored.column_names(), DEFAULT_COLUMNS)
        self.assertEqual(len(restored.get_column("Todo").cards), 1)
        self.assertEqual(restored.get_column("Todo").cards[0].tags, ["x"])

    def test_board_from_dict_missing_name(self):
        with self.assertRaises(ValidationError):
            Board.from_dict({"columns": []})

    def test_all_cards(self):
        board = Board.create_default()
        board.add_card("A", "Todo")
        board.add_card("B", "Done")
        self.assertEqual(len(board.all_cards()), 2)


if __name__ == "__main__":
    unittest.main()
