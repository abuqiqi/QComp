import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen3_tn.evaluation import EvaluationConfig, MetricSpec
from qwen3_tn.evaluation_suite import (
    EvaluationSuiteConfig,
    build_suite_summary,
    render_suite_markdown,
)


class EvaluationSuiteTests(unittest.TestCase):
    def _config(self, root: Path) -> EvaluationSuiteConfig:
        return EvaluationSuiteConfig.from_dict(
            {
                "model_path": root / "model",
                "output_root": root / "results",
                "tasks": [{"name": "task", "evaluation": root / "task.json"}],
                "variants": [
                    {"name": "baseline"},
                    {"name": "mpo", "module_set": root / "before.json"},
                    {"name": "retrained", "module_set": root / "after.json"},
                ],
                "comparison": {
                    "baseline": "baseline",
                    "compressed": "mpo",
                    "retrained": "retrained",
                },
            }
        )

    def test_summary_computes_deltas_and_recovery(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            evaluation = EvaluationConfig(
                "dummy", (MetricSpec("acc,none", "higher"),), limit=1
            )
            for variant, score in (
                ("baseline", 0.8),
                ("mpo", 0.5),
                ("retrained", 0.7),
            ):
                target = config.output_root / variant / "task.json"
                target.parent.mkdir(parents=True)
                target.write_text(
                    json.dumps(
                        {"raw_results": {"results": {"dummy": {"acc,none": score}}}}
                    ),
                    encoding="utf-8",
                )
            summary = build_suite_summary(
                config, {"task": evaluation}, started_at="now", completed_at="later"
            )
            row = summary["comparisons"][0]
            self.assertTrue(summary["complete"])
            self.assertAlmostEqual(row["compressed_delta"], -0.3)
            self.assertAlmostEqual(row["healing_gain"], 0.2)
            self.assertAlmostEqual(row["recovery_fraction"], 2 / 3)
            self.assertIn("66.7%", render_suite_markdown(summary))

    def test_comparison_rejects_unknown_variant(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = {
                "model_path": root / "model",
                "output_root": root / "results",
                "tasks": [{"name": "task", "evaluation": root / "task.json"}],
                "variants": [{"name": "baseline"}],
                "comparison": {
                    "baseline": "baseline",
                    "compressed": "missing",
                    "retrained": "also-missing",
                },
            }
            with self.assertRaisesRegex(ValueError, "unknown variants"):
                EvaluationSuiteConfig.from_dict(value)


if __name__ == "__main__":
    unittest.main()
