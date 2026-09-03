import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from qwen3_tn.evaluation import EvaluationConfig, MetricSpec, extract_metrics
from qwen3_tn.workflows.evaluate import evaluate_with_cache, evaluation_cache_metadata
from helpers import TinyCausalLM, TinyTokenizer, fake_evaluator, make_module_set


class EvaluationTests(unittest.TestCase):
    def test_extract_and_cache(self):
        config = EvaluationConfig("dummy", (MetricSpec("loss", "lower"),), limit=1)
        result = fake_evaluator(TinyCausalLM(), config, tokenizer=TinyTokenizer())
        self.assertIn("loss", extract_metrics(result, config))
        with TemporaryDirectory() as tmp:
            calls = []

            def evaluator(*args, **kwargs):
                calls.append(1)
                return fake_evaluator(*args, **kwargs)

            metadata = evaluation_cache_metadata(
                variant="dense", model_signature={"sha256": "x"}, config=config
            )
            path = Path(tmp) / "eval.json"
            first = evaluate_with_cache(
                TinyCausalLM(),
                TinyTokenizer(),
                config,
                path,
                metadata,
                evaluator=evaluator,
            )
            second = evaluate_with_cache(
                TinyCausalLM(),
                TinyTokenizer(),
                config,
                path,
                metadata,
                evaluator=evaluator,
            )
            self.assertEqual((first.source, second.source), ("evaluation", "cache"))
            self.assertEqual(len(calls), 1)

    def test_tt_cache_identity_includes_version_and_options(self):
        config = EvaluationConfig("dummy", (MetricSpec("loss"),), limit=1)
        with TemporaryDirectory() as tmp:
            index = make_module_set(Path(tmp), TinyCausalLM())
            metadata = evaluation_cache_metadata(
                variant="tt",
                model_signature={"sha256": "x"},
                config=config,
                module_set=index,
                backend="native",
                backend_options={},
            )
            self.assertEqual(
                metadata["tt"]["backend"],
                {"name": "native", "version": "2", "options": {}},
            )
