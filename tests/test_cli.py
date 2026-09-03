import json, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from qwen3_tn.cli.common import (
    load_backend_modules,
    parse_backend_options,
    parse_dtype,
    strict_config,
)


class CLITests(unittest.TestCase):
    def test_strict_config_and_dtype(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"a": 1}))
            self.assertEqual(strict_config(path, allowed={"a"}, required={"a"})["a"], 1)
            with self.assertRaises(ValueError):
                strict_config(path, allowed=set(), required=set())
        self.assertEqual(str(parse_dtype("float32")), "torch.float32")

    def test_backend_module_and_options_parsing(self):
        self.assertEqual(load_backend_modules(["dummy_backend"]), ("dummy_backend",))
        self.assertEqual(parse_backend_options({"x": 1}), {"x": 1})
        with self.assertRaises(ValueError):
            load_backend_modules("dummy_backend")
        with self.assertRaises(ValueError):
            parse_backend_options(["x"])
