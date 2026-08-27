import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from qwen3_tn.benchmarking import (
    LayerBenchmarkConfig,
    TrainingBenchmarkConfig,
    benchmark_tt_layer,
    benchmark_tt_training,
)
from helpers import TinyCausalLM, make_module_set


class BenchmarkTests(unittest.TestCase):
    def test_cpu_layer_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = make_module_set(root, TinyCausalLM())
            result = benchmark_tt_layer(
                LayerBenchmarkConfig(
                    index,
                    "backbone.block.proj",
                    backends=("native",),
                    device="cpu",
                    tokens=2,
                    warmup=0,
                    iterations=1,
                )
            )
            self.assertEqual(result["results"]["native"]["status"], "ok")
            self.assertIn(
                "tokens_per_second", result["results"]["native"]["forward_backward"]
            )

    def test_cpu_training_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = make_module_set(root / "source", TinyCausalLM())
            config = TrainingBenchmarkConfig(
                index,
                backends=("native",),
                device="cpu",
                steps=1,
                warmup_steps=0,
                gradient_accumulation_steps=1,
                gradient_checkpointing=False,
            )

            def batch_factory(_model, device):
                ids = torch.tensor([[1, 2, 3]], device=device)
                return {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone(),
                }

            result = benchmark_tt_training(
                config, TinyCausalLM, batch_factory, output_path=root / "result.json"
            )
            measured = result["results"]["native"]
            self.assertEqual(measured["status"], "ok")
            self.assertEqual(measured["optimizer_step"]["tokens_per_update"], 3)
