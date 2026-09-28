"""Privacy regression tests use synthetic fixtures in temporary repositories."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from NDFitter.paths import PROJECT_ROOT, project_path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_check", ROOT / "scripts/check_repository.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class PathTests(unittest.TestCase):
    def test_relative_path_does_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(project_path("data/example"), str(PROJECT_ROOT / "data/example"))
            finally:
                os.chdir(original)

    def test_explicit_external_path_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="data with spaces ") as directory:
            self.assertEqual(project_path(directory), str(Path(directory).resolve()))


class PolicyTests(unittest.TestCase):
    def test_loader_code_is_allowed_but_datasets_are_not(self):
        self.assertTrue(check.allowed_path("NDFitter/MLP/data/dataset.py"))
        for name in ["NDFitter/MLP/data/points.pickle", "data/readme.md", "NDFitter/GP/data/helper.py", "outputs/result.py", "notes.ipynb", "weights.pt"]:
            with self.subTest(name=name):
                self.assertFalse(check.allowed_path(name))

    def test_detects_private_content_without_echoing_it(self):
        private_path = "/" + "Users" + "/example/project"
        private_email = "person" + "@" + "example.org"
        self.assertEqual(check.content_issue(private_path.encode()), "personal home-directory path")
        self.assertEqual(check.content_issue(private_email.encode()), "email address")
        self.assertEqual(check.content_issue(b"print('safe')\x00"), "binary content")
        self.assertIsNone(check.content_issue(b"print('safe')\n"))


class GitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source only ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", check.IDENTITY[0])
        self.git("config", "user.email", check.IDENTITY[1])
        self.git("config", "commit.gpgsign", "false")
        (self.root / "README.md").write_text("Clean project\n")
        self.git("add", "README.md")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE)

    def scan(self, *args, input=None):
        return subprocess.run(["python3", str(ROOT / "scripts/check_repository.py"), *args], cwd=self.root, input=input, capture_output=True, text=True)

    def test_forced_ignored_data_is_rejected(self):
        (self.root / ".gitignore").write_text("*.pickle\n")
        (self.root / "points.pickle").write_bytes(b"synthetic fixture")
        self.git("add", "-f", "points.pickle")
        result = self.scan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("points.pickle", result.stderr)

    def test_actual_index_is_checked_not_clean_working_file(self):
        (self.root / "README.md").write_text("/" + "home" + "/example/private\n")
        self.git("add", "README.md")
        (self.root / "README.md").write_text("Clean replacement\n")
        self.assertNotEqual(self.scan().returncode, 0)

    def test_deleted_historical_data_still_blocks_push(self):
        (self.root / "points.pickle").write_bytes(b"synthetic fixture")
        self.git("add", "points.pickle")
        self.git("commit", "-qm", "Legacy data")
        self.git("rm", "points.pickle")
        self.git("commit", "-qm", "Remove data")
        self.assertEqual(self.scan().returncode, 0)
        self.assertNotEqual(self.scan("--history").returncode, 0)
        tip = self.git("rev-parse", "HEAD").decode().strip()
        update = f"refs/heads/main {tip} refs/heads/main {'0' * 40}\n"
        self.assertNotEqual(self.scan("--pre-push", input=update).returncode, 0)

    def test_clean_history_passes(self):
        self.git("commit", "-qm", "Clean start")
        self.assertEqual(self.scan("--history").returncode, 0)

    def test_non_anonymous_identity_is_rejected(self):
        self.git("config", "user.name", "Example Person")
        self.git("commit", "-qm", "Initial change")
        self.assertNotEqual(self.scan("--history").returncode, 0)


if __name__ == "__main__":
    unittest.main()
