import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from qwen3_tn.checkpoint import (
    TTModuleSetEntry,
    load_module_set_index,
    load_tt_module,
    save_module_set_index,
    save_tt_module,
)
from qwen3_tn.tt import TTMatrixSpec, tt_svd_matrix


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = TTMatrixSpec((2, 2), (2, 2), (1, 2, 1))
            cores = tt_svd_matrix(torch.randn(4, 4), spec, svd_driver=None)
            manifest = save_tt_module(root / "m", cores, spec, module_path="x.y")
            save_module_set_index(
                root, [TTModuleSetEntry("x.y", "m", manifest.cores_sha256 or "")]
            )
            loaded, saved = load_tt_module(root / "m")
            self.assertEqual(saved.spec, spec)
            torch.testing.assert_close(loaded[0], cores[0])
            _, index = load_module_set_index(root)
            self.assertEqual(index.modules[0].module_path, "x.y")

    def test_corrupt_core_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = TTMatrixSpec((2,), (2,), (1, 1))
            save_tt_module(root, [torch.ones(1, 2, 2, 1)], spec, module_path="x")
            path = root / "cores.safetensors"
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                load_tt_module(root)
