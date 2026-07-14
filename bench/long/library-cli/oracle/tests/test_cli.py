import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import cli


class TestCli(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def run_cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--path", self.path, *args])
        return code, out.getvalue()

    def test_add_list_search(self):
        code, _ = self.run_cli("add", "1", "Dune", "Frank Herbert", "-t", "scifi")
        self.assertEqual(code, 0)
        self.run_cli("add", "2", "Emma", "Jane Austen")
        _, listing = self.run_cli("list")
        self.assertIn("Dune", listing)
        self.assertIn("Emma", listing)
        _, found = self.run_cli("search", "dune")
        self.assertIn("Dune", found)
        self.assertNotIn("Emma", found)
        _, tagged = self.run_cli("search", "-t", "scifi")
        self.assertIn("Dune", tagged)

    def test_checkout_return_cycle(self):
        self.run_cli("add", "1", "Dune", "Frank Herbert")
        code, out = self.run_cli("checkout", "1", "ana")
        self.assertEqual(code, 0)
        self.assertIn("due", out)
        _, status = self.run_cli("search", "dune")
        self.assertIn("[out]", status)
        code, _ = self.run_cli("return", "1")
        self.assertEqual(code, 0)
        _, status = self.run_cli("search", "dune")
        self.assertIn("[in]", status)

    def test_checkout_unknown_isbn_fails(self):
        code, _ = self.run_cli("checkout", "9", "ana")
        self.assertEqual(code, 1)

    def test_remove(self):
        self.run_cli("add", "1", "Dune", "Frank Herbert")
        code, _ = self.run_cli("remove", "1")
        self.assertEqual(code, 0)
        _, listing = self.run_cli("list")
        self.assertNotIn("Dune", listing)

    def test_state_persists_across_invocations(self):
        self.run_cli("add", "1", "Dune", "Frank Herbert")
        self.run_cli("checkout", "1", "ana", "--days", "1")
        _, over = self.run_cli("overdue", "--as-of", "2099-01-01")
        self.assertIn("1", over)


if __name__ == "__main__":
    unittest.main()
