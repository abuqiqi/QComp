import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from qwen3_tn.model import TTTarget
from qwen3_tn.tt import TTMatrixSpec
from qwen3_tn.workflows import RankSweepCandidate, RankSweepConfig, run_rank_sweep
from helpers import TinyCausalLM


class SweepTests(unittest.TestCase):
    def test_baseline_once_and_dense_restored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyCausalLM()
            original = model.backbone.block.proj
            path = "backbone.block.proj"
            target = TTTarget(path, TTMatrixSpec((2, 4), (2, 4), (1, 2, 1)))
            calls = []

            def evaluator(candidate):
                calls.append(1)
                return {"score": float(sum(p.numel() for p in candidate.parameters()))}

            candidates = (
                RankSweepCandidate("r1", {path: (1, 1, 1)}),
                RankSweepCandidate("r2", {path: (1, 2, 1)}),
            )
            config = RankSweepConfig(
                "tiny",
                root / "sweep",
                root / "cache",
                root / "result.json",
                candidates,
                core_dtype=torch.float32,
                svd_driver=None,
                selection_metric="score",
            )
            result = run_rank_sweep(model, [target], config, evaluator)
            self.assertEqual(len(calls), 3)
            self.assertIs(model.backbone.block.proj, original)
            self.assertIn(result["selection"]["candidate"], {"r1", "r2"})
