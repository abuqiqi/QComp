import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from torch import nn

import qwen3_tn.model as model_module
from qwen3_tn.backends import TTLinearBase
from qwen3_tn import import_tt_backend_modules
from qwen3_tn.model import (
    TTModulePatch,
    export_tt_modules,
    find_tt_modules,
    install_tt_modules,
    load_tt_cores,
)
from qwen3_tn.training import freeze_except_tt
from helpers import TinyCausalLM, make_module_set


class ModelTests(unittest.TestCase):
    def test_replacement_transaction_rolls_back_partial_install(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        originals = {"0": model[0], "1": model[1]}
        replacements = {"0": nn.ReLU(), "1": nn.ReLU()}
        replace_module = model_module._replace_module

        def fail_second_install(model, path, module):
            if path == "1" and module is replacements["1"]:
                raise RuntimeError("install failed")
            replace_module(model, path, module)

        with patch.object(model_module, "_replace_module", fail_second_install):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                model_module._install_replacements(model, originals, replacements)

        self.assertIs(model[0], originals["0"])
        self.assertIs(model[1], originals["1"])

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

    def test_external_backend_checkpoint_round_trip(self):
        import_tt_backend_modules(("dummy_backend",))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_index = make_module_set(root / "source", TinyCausalLM())
            first = TinyCausalLM()
            install_tt_modules(
                first,
                source_index,
                tt_backend="test_external",
                trainable=True,
                backend_options={"marker": "checkpoint"},
            )
            exported = export_tt_modules(first, root / "exported")
            second = TinyCausalLM()
            install_tt_modules(
                second,
                exported,
                tt_backend="test_external",
                trainable=True,
                backend_options={"marker": "checkpoint"},
            )
            load_tt_cores(second, exported, require_same_backend=True)
            self.assertEqual(
                set(find_tt_modules(second)), {"backbone.block.proj"}
            )
