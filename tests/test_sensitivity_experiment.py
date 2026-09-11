"""验证敏感性上层入口的资源组装、逐层计划与输出。

本模块用小型 CPU 模型执行 native 压缩，mock 模型加载与 lm-eval 评测，检查筛选分段、
运行配置、模型恢复及报告和标准 JSON 输出，不依赖模型下载。

主要内容：
- ``SensitivityExperimentTests``：验证高层实验与底层分析的连接。
"""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import qcomp.workflows.sensitivity as workflow
from qcomp import (
    CompressionExecutionConfig,
    CompressionTarget,
    ModelLoadConfig,
    MPOSpec,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
)
from qcomp.evaluation import (
    EvaluationResult,
    EvaluationTask,
    EvaluationTaskConfig,
    LMEvalConfig,
)


class SensitivityExperimentTests(unittest.TestCase):
    """检查上层接口的公共行为。"""

    def test_run_selects_slices_restores_and_records(self) -> None:
        """先筛选再分段，执行真实压缩并恢复；结果保留配置且区分两次运行。"""

        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
        )
        with torch.no_grad():
            for layer in model:
                layer.weight.copy_(torch.eye(2))
        original = model[2]
        evaluation = EvaluationResult(
            task=EvaluationTask(
                name="test",
                dataset="synthetic",
                split="test",
                preprocessing="fixed",
                requested_metrics=("acc",),
            ),
            metrics={"acc": 0.75},
            evaluated_examples=1,
        )

        def target(path, linear):
            """为路径 path 对应的 linear 构造满秩单核目标。"""

            return CompressionTarget(
                path,
                "mpo",
                MPOSpec.full_rank((linear.out_features,), (linear.in_features,)),
            )

        import qcomp.workflows.evaluate as engine

        original_compress = engine.compress_model
        expected_random_state = None

        def evaluate_with_random_state(model):
            """模拟评测留下的随机状态，敏感度分解应直接沿用。"""
            nonlocal expected_random_state
            torch.manual_seed(17)
            torch.rand(7)
            expected_random_state = torch.get_rng_state().clone()
            return evaluation

        def compress_with_random_state(*args, **kwargs):
            """确认没有插入独立分解种子，再执行真实压缩。"""
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_random_state))
            return original_compress(*args, **kwargs)

        with TemporaryDirectory() as directory:
            config = SensitivityExperimentConfig(
                name="test/task",
                evaluation=EvaluationTaskConfig(
                    LMEvalConfig(task="test", evaluation_seed=17, num_fewshot=2),
                    {"acc": "higher"},
                ),
                model=ModelLoadConfig(device="cpu", dtype=torch.float32),
                compression=CompressionExecutionConfig(
                    decomposition_provider="native", execution_provider="native"
                ),
                start_layer_index=1,
                max_layers=1,
                artifact_root=directory,
            )
            with (
                patch.object(workflow, "datetime") as clock,
                patch.object(
                    workflow,
                    "load_runtime_config",
                    return_value=SimpleNamespace(
                        model_name_or_path="/models/default", offline=True
                    ),
                ),
                patch.object(
                    workflow,
                    "load_causal_lm",
                    return_value=SimpleNamespace(model=model, tokenizer=object()),
                ) as load,
                patch.object(
                    workflow, "LMEvalEvaluator", return_value=evaluate_with_random_state
                ) as evaluator,
                patch.object(
                    engine, "compress_model", side_effect=compress_with_random_state
                ),
            ):
                clock.now.side_effect = [
                    datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC),
                    datetime(2026, 9, 5, 12, 0, 1, tzinfo=UTC),
                ]
                result = run_sensitivity_experiment(
                    config,
                    select_linear=lambda path, linear: path != "0",
                    make_target=target,
                )
                run_sensitivity_experiment(
                    replace(
                        config,
                        model=replace(config.model, model_name_or_path="custom/model"),
                    ),
                    select_linear=lambda path, linear: path != "0",
                    make_target=target,
                )
            self.assertIs(model[2], original)
            self.assertEqual(len(result.evaluation.plan_results), 1)
            self.assertEqual(
                result.evaluation.plan_results[0].plan.targets[0].module_path, "2"
            )
            self.assertEqual(
                load.call_args_list[0].args[0].model_name_or_path, "/models/default"
            )
            self.assertEqual(
                load.call_args_list[1].args[0].model_name_or_path, "custom/model"
            )
            self.assertIs(evaluator.call_args.args[1], config.evaluation.evaluation)
            root = Path(directory)
            self.assertEqual({p.name for p in root.iterdir()}, {"sensitivity"})
            self.assertEqual(
                {p.name for p in (root / "sensitivity").iterdir()}, {"test-task"}
            )
            run_dirs = sorted((root / "sensitivity/test-task").iterdir())
            self.assertEqual(len(run_dirs), 2)
            self.assertEqual(result.report_path, run_dirs[0] / "layers-001-001.md")
            records = []
            for run_dir in run_dirs:
                self.assertRegex(run_dir.name, r"^\d{8}T\d{6}$")
                report = run_dir / "layers-001-001.md"
                self.assertIn("Sensitivity Report", report.read_text())
                document = json.loads(
                    (run_dir / "sensitivity_results.json").read_text()
                )
                self.assertEqual(document["status"], "completed")
                self.assertEqual(document["expected_cases"], ["2"])
                self.assertEqual(document["provenance"]["start_layer_index"], 1)
                self.assertEqual(
                    document["cases"][0]["compression_plan"]["targets"][0]["spec"][
                        "ranks"
                    ],
                    [1, 1],
                )
                self.assertEqual(document["model"]["dense_parameters"], 12)
                self.assertFalse(list(run_dir.glob("*.jsonl")))
                records.append(document)
            self.assertNotEqual(records[0]["run_id"], records[1]["run_id"])
            self.assertEqual(
                records[0]["evaluation_config"]["evaluation"]["evaluation_seed"], 17
            )
            self.assertEqual(
                records[0]["evaluation_config"]["evaluation"]["num_fewshot"], 2
            )
            self.assertEqual(
                records[0]["execution"]["decomposition_provider"], "native"
            )
            self.assertEqual(records[0]["execution"]["model_dtype"], "torch.float32")
            self.assertEqual(records[0]["model"]["name_or_path"], "/models/default")

    def test_invalid_config_and_empty_selection(self) -> None:
        """非法范围在加载前失败，空选择不进入分析。"""

        base = SensitivityExperimentConfig(
            name="test",
            evaluation=EvaluationTaskConfig(
                LMEvalConfig(task="test"), {"acc": "higher"}
            ),
        )
        for values in (
            {"start_layer_index": -1},
            {"max_layers": 0},
            {"name": " "},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(base, **values)
        with self.assertRaises(ValueError):
            CompressionExecutionConfig(decomposition_dtype=torch.int32)
        with patch.object(workflow, "load_causal_lm") as load:
            with self.assertRaises(ValueError):
                run_sensitivity_experiment(
                    replace(
                        base,
                        evaluation=EvaluationTaskConfig(
                            base.evaluation.evaluation, {"acc": "invalid"}
                        ),
                    ),
                    select_linear=lambda p, l: True,
                    make_target=lambda p, l: None,
                )
            load.assert_not_called()
        with (
            patch.object(
                workflow,
                "load_runtime_config",
                return_value=SimpleNamespace(model_name_or_path="model"),
            ),
            patch.object(
                workflow,
                "load_causal_lm",
                return_value=SimpleNamespace(model=nn.Sequential(), tokenizer=None),
            ),
            patch.object(workflow, "evaluate_compression_plans") as analyze,
        ):
            with self.assertRaisesRegex(ValueError, "empty"):
                run_sensitivity_experiment(
                    base, select_linear=lambda p, l: True, make_target=lambda p, l: None
                )
            analyze.assert_not_called()
