"""Unit tests for every cli.py subcommand, including tag filtering edge cases."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import cli


class CLITestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", prefix="kan_cli_"
        )
        self.tmp.close()
        os.remove(self.tmp.name)
        self.path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def run_cli(self, *args):
        """Run the CLI with the given args, returning (stdout, stderr, exit)."""
        argv = ["--path", self.path] + list(args)
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = cli.main(argv)
            except SystemExit as exc:
                code = exc.code if exc.code is not None else 0
        return out.getvalue(), err.getvalue(), code

    def init_board(self, name="Test Board"):
        return self.run_cli("init", "--name", name)


class TestInitCommand(CLITestBase):
    def test_init_creates_default_board(self):
        out, err, code = self.init_board()
        self.assertEqual(code, 0, err)
        self.assertIn("Created board 'Test Board'", out)
        self.assertIn("Todo", out)
        self.assertIn("In Progress", out)
        self.assertIn("Done", out)
        self.assertTrue(os.path.exists(self.path))

    def test_init_refuses_existing_without_force(self):
        self.init_board()
        out, err, code = self.run_cli("init", "--name", "Other")
        self.assertNotEqual(code, 0)
        self.assertIn("already exists", err)

    def test_init_force_overwrites(self):
        self.init_board("First")
        out, err, code = self.run_cli("init", "--name", "Second", "--force")
        self.assertEqual(code, 0, err)
        self.assertIn("Second", out)


class TestAddCommand(CLITestBase):
    def setUp(self):
        super().setUp()
        self.init_board()

    def test_add_basic(self):
        out, err, code = self.run_cli("add", "My Task")
        self.assertEqual(code, 0, err)
        self.assertIn("Added card 'My Task'", out)
        self.assertIn("Todo", out)

    def test_add_with_column_description_tags(self):
        out, err, code = self.run_cli(
            "add", "Task", "-c", "In Progress", "-d", "a desc", "-t", "py", "-t", "api"
        )
        self.assertEqual(code, 0, err)
        self.assertIn("[api, py]", out)

    def test_add_comma_separated_tags(self):
        out, err, code = self.run_cli("add", "Task", "-t", "a,b,c")
        self.assertEqual(code, 0, err)
        self.assertIn("[a, b, c]", out)

    def test_add_unknown_column_fails(self):
        out, err, code = self.run_cli("add", "Task", "-c", "Nowhere")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown column", err)

    def test_add_empty_title_fails(self):
        out, err, code = self.run_cli("add", "   ")
        self.assertNotEqual(code, 0)

    def test_add_invalid_tag_fails(self):
        out, err, code = self.run_cli("add", "Task", "-t", "bad tag!")
        self.assertNotEqual(code, 0)
        self.assertIn("tag", err)


class TestMoveCommand(CLITestBase):
    def setUp(self):
        super().setUp()
        self.init_board()
        self.run_cli("add", "Task One", "-c", "Todo")
        self.run_cli("add", "Task Two", "-c", "Todo")

    def test_move_by_id(self):
        from storage import load_board
        board = load_board(self.path)
        card_id = board.get_column("Todo").cards[0].id
        out, err, code = self.run_cli("move", card_id, "In Progress")
        self.assertEqual(code, 0, err)
        self.assertIn("Moved card", out)

    def test_move_by_title(self):
        out, err, code = self.run_cli("move", "task one", "Done")
        self.assertEqual(code, 0, err)
        self.assertIn("Moved card 'Task One'", out)

    def test_move_unknown_card_fails(self):
        out, err, code = self.run_cli("move", "nope", "Done")
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)

    def test_move_unknown_target_fails(self):
        out, err, code = self.run_cli("move", "Task One", "Nowhere")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown column", err)

    def test_move_done_back_to_todo_forbidden(self):
        self.run_cli("move", "Task One", "Done")
        out, err, code = self.run_cli("move", "Task One", "Todo")
        self.assertNotEqual(code, 0)
        self.assertIn("Done", err)


class TestListCommand(CLITestBase):
    def setUp(self):
        super().setUp()
        self.init_board()
        self.run_cli("add", "Alpha", "-c", "Todo", "-t", "frontend", "-t", "urgent")
        self.run_cli("add", "Beta", "-c", "Todo", "-t", "backend")
        self.run_cli("add", "Gamma", "-c", "In Progress", "-t", "frontend")

    def test_list_shows_all_cards(self):
        out, err, code = self.run_cli("list")
        self.assertEqual(code, 0, err)
        self.assertIn("Alpha", out)
        self.assertIn("Beta", out)
        self.assertIn("Gamma", out)
        self.assertIn("Total cards: 3", out)

    def test_list_filter_single_tag(self):
        out, err, code = self.run_cli("list", "-t", "frontend")
        self.assertEqual(code, 0, err)
        self.assertIn("Alpha", out)
        self.assertIn("Gamma", out)
        self.assertNotIn("Beta", out)
        self.assertIn("Total cards: 2", out)

    def test_list_filter_tag_case_insensitive(self):
        out, err, code = self.run_cli("list", "-t", "FRONTEND")
        self.assertEqual(code, 0, err)
        self.assertIn("Alpha", out)
        self.assertIn("Total cards: 2", out)

    def test_list_filter_multiple_tags_union(self):
        out, err, code = self.run_cli("list", "-t", "backend", "-t", "urgent")
        self.assertEqual(code, 0, err)
        self.assertIn("Alpha", out)  # urgent
        self.assertIn("Beta", out)   # backend
        self.assertNotIn("Gamma", out)
        self.assertIn("Total cards: 2", out)

    def test_list_filter_comma_separated(self):
        out, err, code = self.run_cli("list", "-t", "backend,urgent")
        self.assertEqual(code, 0, err)
        self.assertIn("Alpha", out)
        self.assertIn("Beta", out)
        self.assertNotIn("Gamma", out)

    def test_list_filter_no_matches(self):
        out, err, code = self.run_cli("list", "-t", "nonexistent")
        self.assertEqual(code, 0, err)
        self.assertIn("(empty)", out)
        self.assertIn("Total cards: 0", out)

    def test_list_empty_board(self):
        # Fresh board with no cards.
        path2 = self.path + ".2"
        try:
            argv = ["--path", path2, "init"]
            cli.main(argv)
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main(["--path", path2, "list"])
            self.assertIn("(empty)", out.getvalue())
            self.assertIn("Total cards: 0", out.getvalue())
        finally:
            if os.path.exists(path2):
                os.remove(path2)


class TestDeleteCommand(CLITestBase):
    def setUp(self):
        super().setUp()
        self.init_board()
        self.run_cli("add", "Task One", "-c", "Todo")

    def test_delete_by_title(self):
        out, err, code = self.run_cli("delete", "task one")
        self.assertEqual(code, 0, err)
        self.assertIn("Deleted card 'Task One'", out)

    def test_delete_by_id(self):
        from storage import load_board
        board = load_board(self.path)
        card_id = board.get_column("Todo").cards[0].id
        out, err, code = self.run_cli("delete", card_id)
        self.assertEqual(code, 0, err)
        self.assertIn("Deleted card", out)

    def test_delete_unknown_fails(self):
        out, err, code = self.run_cli("delete", "nope")
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)

    def test_delete_persists(self):
        self.run_cli("delete", "Task One")
        out, err, code = self.run_cli("list")
        self.assertEqual(code, 0, err)
        self.assertIn("Total cards: 0", out)


class TestStatsCommand(CLITestBase):
    def setUp(self):
        super().setUp()
        self.init_board()
        self.run_cli("add", "Alpha", "-c", "Todo", "-t", "frontend")
        self.run_cli("add", "Beta", "-c", "Todo", "-t", "backend")
        self.run_cli("add", "Gamma", "-c", "Done", "-t", "frontend")

    def test_stats_counts(self):
        out, err, code = self.run_cli("stats")
        self.assertEqual(code, 0, err)
        self.assertIn("Cards per column", out)
        self.assertIn("Todo", out)
        self.assertIn("TOTAL", out)
        self.assertIn("Tag distribution", out)
        self.assertIn("frontend", out)
        self.assertIn("backend", out)

    def test_stats_empty_board(self):
        path2 = self.path + ".2"
        try:
            cli.main(["--path", path2, "init"])
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main(["--path", path2, "stats"])
            text = out.getvalue()
            self.assertIn("TOTAL", text)
            self.assertIn("(no tags)", text)
        finally:
            if os.path.exists(path2):
                os.remove(path2)


class TestNoBoardFile(CLITestBase):
    """Commands that need a board should fail gracefully when none exists."""

    def test_add_without_board(self):
        out, err, code = self.run_cli("add", "Task")
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)

    def test_list_without_board(self):
        out, err, code = self.run_cli("list")
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)

    def test_stats_without_board(self):
        out, err, code = self.run_cli("stats")
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)


if __name__ == "__main__":
    unittest.main()
