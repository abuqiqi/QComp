import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from qwen3_tn.data import prepare_causal_lm_data
from qwen3_tn.evaluation import EvaluationConfig, MetricSpec
from qwen3_tn.model import TTTarget
from qwen3_tn.training import TTFineTuneConfig
from qwen3_tn.tt import TTMatrixSpec
from qwen3_tn.workflows import (
    DecomposeConfig,
    FineTuneExperimentConfig,
    decompose_targets,
    run_finetune_experiment,
)
from helpers import TinyCausalLM, TinyTokenizer, fake_evaluator, make_module_set


class Source:
    def documents(self):
        return iter(("abc", "def"))

    def fingerprint(self):
        return "source-v1"

    def metadata(self):
        return {"type": "test"}


class WorkflowTests(unittest.TestCase):
    def test_decompose_is_model_agnostic(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyCausalLM()
            target = TTTarget(
                "backbone.block.proj", TTMatrixSpec((2, 4), (2, 4), (1, 2, 1)), 2
            )
            result = decompose_targets(
                model,
                [target],
                DecomposeConfig(
                    "tiny",
                    root / "out",
                    root / "cache",
                    None,
                    torch.float32,
                    torch.float32,
                ),
            )
            self.assertTrue((root / "out" / "index.json").is_file())
            self.assertEqual(result["aggregate"]["target_count"], 1)

    def test_finetune_and_reuse(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            model = TinyCausalLM()
            module_set = make_module_set(root / "source", model)
            tokenizer = TinyTokenizer()
            prepared = prepare_causal_lm_data(
                Source(), tokenizer, max_length=4, batch_size=1
            )
            evaluation = EvaluationConfig("dummy", (MetricSpec("loss"),), limit=1)
            training = TTFineTuneConfig(
                max_steps=1,
                gradient_accumulation_steps=1,
                device="cpu",
                bf16_autocast=False,
                logging_steps=1,
                save_steps=0,
            )
            config = FineTuneExperimentConfig(
                model_dir,
                module_set,
                root / "artifacts",
                root / "results",
                training,
                evaluation,
            )
            first = run_finetune_experiment(
                config,
                prepared,
                model=model,
                tokenizer=tokenizer,
                evaluator=fake_evaluator,
            )
            run_path = config.artifact_root / "run.json"
            run = json.loads(run_path.read_text())
            run["training_signature"]["backend"] = "native"
            run_path.write_text(json.dumps(run))
            second = run_finetune_experiment(
                config,
                prepared,
                model=TinyCausalLM(),
                tokenizer=tokenizer,
                evaluator=fake_evaluator,
            )
            self.assertEqual(first["training_source"], "training")
            self.assertEqual(second["training_source"], "checkpoint")
