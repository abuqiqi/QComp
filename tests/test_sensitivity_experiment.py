"""验证敏感性上层入口的资源组装、逐层计划与输出。

本模块用小型 CPU 模型执行 native 压缩，mock 模型加载与 lm-eval 评测，检查筛选分段、
运行配置、模型恢复及报告和 JSON Lines 输出，不依赖模型下载。

主要内容：
- ``SensitivityExperimentTests``：验证高层实验与底层分析的连接。
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from qcomp import (
    CompressionTarget,
    MPOSpec,
    ModelLoadConfig,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
)
from qcomp.evaluation import EvaluationResult, EvaluationTask, LMEvalConfig
import qcomp.workflows.sensitivity as workflow


class SensitivityExperimentTests(unittest.TestCase):
    """检查上层接口的公共行为。"""

    def test_run_selects_slices_restores_and_records(self) -> None:
        """先筛选再分段，执行真实压缩并恢复；日志保留配置且区分两次运行。"""

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

        with TemporaryDirectory() as directory:
            config = SensitivityExperimentConfig(
                name="test/task",
                evaluation=LMEvalConfig(task="test", seed=17, num_fewshot=2),
                metrics=("acc",),
                model=ModelLoadConfig(device="cpu", dtype=torch.float32),
                decomposition_provider="native",
                execution_provider="native",
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
                    workflow, "LMEvalEvaluator", return_value=lambda model: evaluation
                ) as evaluator,
            ):
                clock.now.side_effect = [
                    datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2026, 9, 5, 12, 0, 1, tzinfo=timezone.utc),
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
            self.assertIs(evaluator.call_args.args[1], config.evaluation)
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
                run_records = [
                    json.loads(line)
                    for line in report.with_suffix(".jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(run_records), 3)
                self.assertEqual(run_records[0]["fields"]["start_layer_index"], 1)
                self.assertNotIn("start_index", run_records[0]["fields"])
                records.extend(run_records)
            self.assertEqual(
                [r["event"] for r in records],
                ["experiment_started", "case_completed", "experiment_completed"] * 2,
            )
            self.assertNotEqual(
                records[0]["fields"]["run_id"], records[3]["fields"]["run_id"]
            )
            for offset in (0, 3):
                self.assertEqual(
                    len({r["fields"]["run_id"] for r in records[offset : offset + 3]}),
                    1,
                )
            fields = records[0]["fields"]
            for key, value in {
                "seed": 17,
                "num_fewshot": 2,
                "decomposition_provider": "native",
                "model_dtype": "torch.float32",
                "model_name_or_path": "/models/default",
            }.items():
                self.assertEqual(fields[key], value)

    def test_invalid_config_and_empty_selection(self) -> None:
        """非法范围在加载前失败，空选择不进入分析。"""

        base = SensitivityExperimentConfig(
            name="test", evaluation=LMEvalConfig(task="test"), metrics=("acc",)
        )
        for values in (
            {"start_layer_index": -1},
            {"max_layers": 0},
            {"name": " "},
            {"decomposition_dtype": torch.int32},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(base, **values)
        with patch.object(workflow, "load_causal_lm") as load:
            with self.assertRaises(ValueError):
                run_sensitivity_experiment(
                    replace(base, metrics=("unknown_metric",)),
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
