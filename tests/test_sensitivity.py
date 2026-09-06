"""验证通用压缩敏感性分析工作流。

本模块使用小型无 bias Linear 模型、native MPO backend 和确定性 evaluator，检查
baseline、多层压缩方案、多指标退化计算、逐层与整模压缩指标以及所有恢复路径。测试
只验证 workflow 编排，不依赖真实数据集或具体模型评测库。

主要内容：
- ``SensitivityTests``：验证敏感性分析的正常流程、输入检查和异常恢复。
- ``_evaluation``：根据当前安装的张量网络层数量生成确定性多指标结果。
"""

import inspect
import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

import qcomp.workflows.sensitivity as sensitivity_module
from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    SensitivityCase,
    analyze_sensitivity,
    format_sensitivity_report,
    sensitivity_case_record,
    get_backend,
    list_tensor_network_linears,
)
from qcomp.evaluation import EvaluationResult, EvaluationTask


_TASK = EvaluationTask(
    name="test-quality",
    dataset="synthetic",
    split="test",
    preprocessing="fixed",
    requested_metrics=("accuracy", "loss", "auxiliary"),
)


def _evaluation(model: nn.Module) -> EvaluationResult:
    """根据当前压缩层数量返回确定性的多个评测指标。

    参数：
        model: 当前处于未压缩或临时压缩状态的测试模型。

    返回：
        包含 accuracy、loss 和额外指标的通用评测结果。
    """

    compressed_layers = len(list_tensor_network_linears(model))
    return EvaluationResult(
        task=_TASK,
        metrics={
            "accuracy": 0.8 - 0.1 * compressed_layers,
            "loss": 1.0 + 0.2 * compressed_layers,
            "auxiliary": 10.0 + compressed_layers,
        },
        evaluated_examples=4,
    )


class SensitivityTests(unittest.TestCase):
    """验证通用敏感性分析的数据流和模型恢复行为。"""

    def setUp(self) -> None:
        """创建双层测试模型、MPO 配置和外部 native backend。"""

        torch.manual_seed(41)
        self.model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        ).to(torch.float64)
        self.originals = (self.model[0], self.model[1])
        self.spec = MPOSpec.full_rank((2, 2), (2, 2))
        self.backend = get_backend("native", "mpo")

    def _case(self, name: str, *paths: str) -> SensitivityCase:
        """为指定层路径创建一个测试用 MPO 敏感性实验。

        参数：
            name: 实验名称。
            paths: 当前实验同时压缩的模块路径。

        返回：
            包含一个或多个目标的敏感性实验。
        """

        return SensitivityCase(
            name=name,
            compression_plan=CompressionPlan(
                tuple(
                    CompressionTarget(path, "mpo", self.spec)
                    for path in paths
                )
            ),
        )

    def test_case_record_contains_independent_serializable_fields(self) -> None:
        """结果转换保留指标和耗时，生成可序列化且不共享可变字段的记录。"""

        result = SimpleNamespace(
            case=SimpleNamespace(name="layer-0"),
            evaluation=SimpleNamespace(metrics={"acc": 0.75}),
            metric_degradations={"acc": 0.05},
            layer_compressions={
                "block": SimpleNamespace(compression_ratio=2.0, relative_error=0.1)
            },
            model_compression=SimpleNamespace(compression_ratio=1.5),
            compression_seconds=2.5,
            evaluation_seconds=3.5,
        )
        record = sensitivity_case_record(result)
        self.assertEqual(json.loads(json.dumps(record)), {
            "case": "layer-0",
            "metrics": {"acc": 0.75},
            "degradations": {"acc": 0.05},
            "layers": {"block": {"compression_ratio": 2.0, "relative_error": 0.1}},
            "model_ratio": 1.5,
            "compression_seconds": 2.5,
            "evaluation_seconds": 3.5,
        })
        record["metrics"]["acc"] = 0.0
        record["degradations"]["acc"] = 0.0
        record["layers"]["block"]["relative_error"] = 0.0
        self.assertEqual(result.evaluation.metrics["acc"], 0.75)
        self.assertEqual(result.metric_degradations["acc"], 0.05)
        self.assertEqual(result.layer_compressions["block"].relative_error, 0.1)

    def test_float32_decomposition_preserves_bfloat16_model(self) -> None:
        """验证精度参数贯穿敏感性工作流，压缩层可运行且原始权重被恢复。"""

        self.model.to(torch.bfloat16)
        inputs = torch.ones(2, 4, dtype=torch.bfloat16)
        expected = self.model(inputs).detach().clone()
        weights = tuple(layer.weight.detach().clone() for layer in self.originals)

        def evaluator(model: nn.Module) -> EvaluationResult:
            """运行当前模型并检查精度和数值，参数 model 为当前评测模型。"""

            self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
            torch.testing.assert_close(model(inputs), expected, rtol=0.05, atol=0.005)
            return _evaluation(model)

        with patch.object(
            self.backend, "decompose", wraps=self.backend.decompose
        ) as decompose:
            analyze_sensitivity(
                self.model,
                (self._case("bf16", "0", "1"),),
                decomposition_backends={"mpo": self.backend},
                execution_backends={"mpo": self.backend},
                decomposition_dtype=torch.float32,
                evaluator=evaluator,
                metric_directions={"accuracy": "higher"},
            )
        self.assertEqual(decompose.call_count, 2)
        for call in decompose.call_args_list:
            self.assertEqual(call.args[0].dtype, torch.float32)
        for index, original in enumerate(self.originals):
            self.assertIs(self.model[index], original)
            torch.testing.assert_close(original.weight, weights[index], rtol=0, atol=0)

    def test_analyze_single_and_multiple_layer_cases(self) -> None:
        """复用外部 backend 评测单层与多层方案并返回多指标退化。"""

        calls = 0

        def evaluator(model: nn.Module) -> EvaluationResult:
            """记录评测次数并返回确定性结果。

            参数：
                model: 当前测试模型。

            返回：
                根据压缩层数量生成的评测结果。
            """

            nonlocal calls
            calls += 1
            return _evaluation(model)

        result = analyze_sensitivity(
            self.model,
            (
                self._case("first-low", "0"),
                self._case("both-medium", "0", "1"),
            ),
            decomposition_backends={"mpo": self.backend},
            execution_backends={"mpo": self.backend},
            evaluator=evaluator,
            metric_directions={
                "accuracy": "higher",
                "loss": "lower",
            },
        )

        self.assertEqual(calls, 3)
        self.assertEqual(result.baseline_evaluation.metrics["auxiliary"], 10.0)
        self.assertGreaterEqual(result.baseline_evaluation_seconds, 0.0)
        first, both = result.case_results
        self.assertAlmostEqual(first.metric_degradations["accuracy"], 0.1)
        self.assertAlmostEqual(first.metric_degradations["loss"], 0.2)
        self.assertNotIn("auxiliary", first.metric_degradations)
        self.assertEqual(tuple(first.layer_compressions), ("0",))
        self.assertEqual(set(both.layer_compressions), {"0", "1"})
        self.assertEqual(first.model_compression.compressed_layers, 1)
        self.assertEqual(both.model_compression.compressed_layers, 2)
        self.assertGreaterEqual(first.compression_seconds, 0.0)
        self.assertGreaterEqual(first.evaluation_seconds, 0.0)
        self.assertIs(self.model[0], self.originals[0])
        self.assertIs(self.model[1], self.originals[1])

    def test_case_callback_runs_after_model_restoration(self) -> None:
        """每个 case 恢复原始 Linear 后立即把结果交给回调。"""

        observed = []

        def on_case_result(case_result) -> None:
            """记录回调结果并确认模型已经恢复。

            参数：
                case_result: 刚完成的单层敏感性结果。
            """

            self.assertIs(self.model[0], self.originals[0])
            self.assertIs(self.model[1], self.originals[1])
            observed.append(case_result)

        result = analyze_sensitivity(
            self.model,
            (self._case("first-live", "0"),),
            decomposition_backends={"mpo": self.backend},
            execution_backends={"mpo": self.backend},
            evaluator=_evaluation,
            metric_directions={"accuracy": "higher"},
            on_case_result=on_case_result,
        )

        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], result.case_results[0])

    def test_format_report_returns_and_writes_same_markdown(self) -> None:
        """格式化多指标结果，并把相同 Markdown 写入指定路径。"""

        result = analyze_sensitivity(
            self.model,
            (self._case("first-low", "0"),),
            decomposition_backends={"mpo": self.backend},
            execution_backends={"mpo": self.backend},
            evaluator=_evaluation,
            metric_directions={"accuracy": "higher", "loss": "lower"},
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "reports" / "sensitivity.md"
            report = format_sensitivity_report(result, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), report)

        self.assertEqual(format_sensitivity_report(result), report)
        self.assertIn("# Sensitivity Report", report)
        self.assertIn("## Baseline Metrics", report)
        self.assertIn("## Cases", report)
        self.assertIn("## Metrics", report)
        self.assertIn("## Layers", report)
        self.assertIn("first-low", report)
        self.assertIn("accuracy", report)
        self.assertIn("loss", report)
        self.assertIn("| first-low | auxiliary | 11 |  |", report)
        self.assertIn("mpo", report)

    def test_evaluation_failure_restores_model(self) -> None:
        """压缩模型评测失败时恢复该实验替换的全部 Linear。"""

        calls = 0

        def evaluator(model: nn.Module) -> EvaluationResult:
            """返回 baseline，并在压缩状态下模拟评测失败。

            参数：
                model: 当前测试模型。

            返回：
                未压缩状态的基线结果。

            异常：
                RuntimeError: 模型包含压缩层时抛出。
            """

            nonlocal calls
            calls += 1
            if list_tensor_network_linears(model):
                raise RuntimeError("planned evaluation failure")
            return _evaluation(model)

        with self.assertRaisesRegex(RuntimeError, "planned evaluation failure"):
            analyze_sensitivity(
                self.model,
                (self._case("both", "0", "1"),),
                decomposition_backends={"mpo": self.backend},
                execution_backends={"mpo": self.backend},
                evaluator=evaluator,
                metric_directions={"accuracy": "higher"},
            )

        self.assertEqual(calls, 2)
        self.assertIs(self.model[0], self.originals[0])
        self.assertIs(self.model[1], self.originals[1])

    def test_input_and_metric_validation(self) -> None:
        """拒绝空实验、重复名称、空指标和评测结果缺失指标。"""

        arguments = {
            "decomposition_backends": {"mpo": self.backend},
            "execution_backends": {"mpo": self.backend},
            "evaluator": _evaluation,
        }
        with self.assertRaisesRegex(ValueError, "cases"):
            analyze_sensitivity(
                self.model,
                (),
                metric_directions={"accuracy": "higher"},
                **arguments,
            )
        case = self._case("same", "0")
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_sensitivity(
                self.model,
                (case, case),
                metric_directions={"accuracy": "higher"},
                **arguments,
            )
        with self.assertRaisesRegex(ValueError, "metric_directions"):
            analyze_sensitivity(
                self.model,
                (case,),
                metric_directions={},
                **arguments,
            )
        with self.assertRaisesRegex(ValueError, "does not contain"):
            analyze_sensitivity(
                self.model,
                (case,),
                metric_directions={"perplexity": "lower"},
                **arguments,
            )

    def test_metric_direction_and_evaluation_task_validation(self) -> None:
        """拒绝未知指标方向以及与 baseline 不同的评测任务。"""

        case = self._case("first", "0")
        arguments = {
            "decomposition_backends": {"mpo": self.backend},
            "execution_backends": {"mpo": self.backend},
        }
        with self.assertRaisesRegex(ValueError, "higher.*lower"):
            analyze_sensitivity(
                self.model,
                (case,),
                evaluator=_evaluation,
                metric_directions={"accuracy": "largest"},
                **arguments,
            )

        calls = 0

        def changing_task_evaluator(model: nn.Module) -> EvaluationResult:
            """让压缩状态的评测返回不同任务。

            参数：
                model: 当前测试模型。

            返回：
                baseline 使用固定任务，压缩状态使用另一任务的评测结果。
            """

            nonlocal calls
            calls += 1
            result = _evaluation(model)
            if calls == 1:
                return result
            return EvaluationResult(
                task=EvaluationTask(
                    name="other",
                    dataset="synthetic",
                    split="test",
                    preprocessing="fixed",
                    requested_metrics=("accuracy",),
                ),
                metrics=result.metrics,
                evaluated_examples=result.evaluated_examples,
            )

        with self.assertRaisesRegex(ValueError, "same EvaluationTask"):
            analyze_sensitivity(
                self.model,
                (case,),
                evaluator=changing_task_evaluator,
                metric_directions={"accuracy": "higher"},
                **arguments,
            )
        self.assertIs(self.model[0], self.originals[0])

    def test_core_analysis_has_no_resource_creation_dependency(self) -> None:
        """确认底层分析不创建 backend、MPO spec 或 lm-eval evaluator。"""

        source = inspect.getsource(sensitivity_module.analyze_sensitivity)
        for forbidden in ("get_backend", "MPOSpec", "tensorly", "LMEvalEvaluator"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
