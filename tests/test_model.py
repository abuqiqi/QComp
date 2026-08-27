import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torch import nn

from qwen3_tn.backends import TTLinearBase
from qwen3_tn.model import TTModulePatch, find_tt_modules, install_tt_modules
from qwen3_tn.training import freeze_except_tt
from helpers import TinyCausalLM, make_module_set


class ModelTests(unittest.TestCase):
    def test_generic_install_and_patch_restore(self):
        with TemporaryDirectory() as tmp:
            model = TinyCausalLM()
            index = make_module_set(Path(tmp), model)
            original = model.backbone.block.proj
            patch = TTModulePatch(model, index, trainable=True)
            for _ in range(2):
                with patch:
                    self.assertIsInstance(model.backbone.block.proj, TTLinearBase)
                    self.assertEqual(
                        set(find_tt_modules(model)), {"backbone.block.proj"}
                    )
                self.assertIs(model.backbone.block.proj, original)

    def test_install_validates_dense_type(self):
        with TemporaryDirectory() as tmp:
            source = TinyCausalLM()
            index = make_module_set(Path(tmp), source)
            target = TinyCausalLM()
            target.backbone.block.proj = nn.ReLU()
            with self.assertRaises(TypeError):
                install_tt_modules(target, index)

    def test_freeze_requires_tt_modules(self):
        with self.assertRaisesRegex(RuntimeError, "no TT"):
            freeze_except_tt(TinyCausalLM())
