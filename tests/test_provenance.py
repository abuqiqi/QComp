import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qwen3_tn.provenance import atomic_publish_directory


class ProvenanceTests(unittest.TestCase):
    def test_atomic_publish_directory_replaces_existing_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            temporary = root / "temporary"
            target = root / "target"
            temporary.mkdir()
            target.mkdir()
            (temporary / "new.txt").write_text("new")
            (target / "old.txt").write_text("old")

            atomic_publish_directory(temporary, target)

            self.assertEqual((target / "new.txt").read_text(), "new")
            self.assertFalse((target / "old.txt").exists())
            self.assertFalse(temporary.exists())

    def test_atomic_publish_directory_restores_target_on_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            temporary = root / "temporary"
            target = root / "target"
            temporary.mkdir()
            target.mkdir()
            (target / "old.txt").write_text("old")
            replace = os.replace
            calls = 0

            def fail_publish(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("publish failed")
                replace(source, destination)

            with patch("qwen3_tn.provenance.os.replace", side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    atomic_publish_directory(temporary, target)

            self.assertEqual((target / "old.txt").read_text(), "old")
            self.assertTrue(temporary.exists())


if __name__ == "__main__":
    unittest.main()
