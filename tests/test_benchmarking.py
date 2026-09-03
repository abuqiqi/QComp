import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from qwen3_tn.benchmarking import (
    InferenceBenchmarkConfig,
    LayerBenchmarkConfig,
    TrainingBenchmarkConfig,
    benchmark_tt_inference,
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
            self.assertIn(
                "peak_allocated_bytes", result["results"]["native"]["forward"]
            )

    def test_external_backend_options_and_full_inference_schema(self):
        from qwen3_tn import import_tt_backend_modules

        import_tt_backend_modules(("dummy_backend",))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = make_module_set(root, TinyCausalLM())
            layer = benchmark_tt_layer(
                LayerBenchmarkConfig(
                    index,
                    "backbone.block.proj",
                    backends=("test_external",),
                    backend_options={"test_external": {"marker": "benchmark"}},
                    device="cpu",
                    tokens=2,
                    warmup=0,
                    iterations=1,
                )
            )
            self.assertEqual(
                layer["results"]["test_external"]["backend_metadata"]["options"],
                {"marker": "benchmark"},
            )
            inference = benchmark_tt_inference(
                InferenceBenchmarkConfig(
                    index,
                    backends=("native",),
                    device="cpu",
                    batch_size=1,
                    prefill_tokens=2,
                    decode_steps=2,
                    warmup=0,
                    iterations=1,
                    core_dtype=torch.float32,
                ),
                TinyCausalLM,
            )
            measured = inference["results"]["native"]
            self.assertEqual(measured["status"], "ok")
            self.assertEqual(measured["module_count"], 1)
            self.assertEqual(
                measured["comparison_vs_native"]["top1_agreement"], 1.0
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
