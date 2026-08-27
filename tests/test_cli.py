import json, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from qwen3_tn.cli.common import parse_dtype, strict_config


class CLITests(unittest.TestCase):
    def test_strict_config_and_dtype(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"a": 1}))
            self.assertEqual(strict_config(path, allowed={"a"}, required={"a"})["a"], 1)
            with self.assertRaises(ValueError):
                strict_config(path, allowed=set(), required=set())
        self.assertEqual(str(parse_dtype("float32")), "torch.float32")
