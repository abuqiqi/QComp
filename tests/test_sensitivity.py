"""验证通用结果转换为既有敏感度日志与 Markdown 的兼容性。

主要内容：
- 独立 JSON 事件字段与不共享可变指标。
- 基线、多指标、逐层误差及报告文件与返回文本一致。
"""

import json
from types import SimpleNamespace

from torch import nn

from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    evaluate_compression_plans,
    format_sensitivity_report,
    get_backend,
    sensitivity_case_record,
)
from qcomp.evaluation import EvaluationTask, EvaluationResult


def test_case_record_serializable_and_independent():
    """转换器保留旧字段，复制指标与误差字典，不修改通用结果。"""
    result = SimpleNamespace(
        name="layer-0",
        evaluations={
            "task": SimpleNamespace(
                evaluation=SimpleNamespace(metrics={"acc": 0.75}), seconds=3.5
            )
        },
        metric_degradations={"task": {"acc": 0.05}},
        layer_compressions={
            "block": SimpleNamespace(compression_ratio=2.0, relative_error=0.1)
        },
        model_compression=SimpleNamespace(compression_ratio=1.5),
        compression_seconds=2.5,
    )
    record = sensitivity_case_record(result, "task")
    assert json.loads(json.dumps(record)) == {
        "case": "layer-0",
        "metrics": {"acc": 0.75},
        "degradations": {"acc": 0.05},
        "layers": {"block": {"compression_ratio": 2.0, "relative_error": 0.1}},
        "model_ratio": 1.5,
        "compression_seconds": 2.5,
        "evaluation_seconds": 3.5,
    }
    record["metrics"]["acc"] = 0
    record["degradations"]["acc"] = 0
    record["layers"]["block"]["relative_error"] = 0
    assert result.evaluations["task"].evaluation.metrics["acc"] == 0.75
    assert result.metric_degradations["task"]["acc"] == 0.05
    assert result.layer_compressions["block"].relative_error == 0.1


def test_report_roundtrip_and_legacy_tables(tmp_path):
    """多个指标和额外指标仍输出原报告表格，文件和返回文本一致。"""
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    backend = get_backend("native", "mpo")
    plan = CompressionPlan(
        (CompressionTarget("0", "mpo", MPOSpec.full_rank((2, 2), (2, 2))),)
    )
    task = EvaluationTask(
        "test", "synthetic", "test", "fixed", ("acc", "loss", "auxiliary")
    )

    def evaluator(current):
        """提供稳定的三指标结果。"""
        return EvaluationResult(
            task, {"acc": 0.8, "loss": 1.0, "auxiliary": 11.0}, 4, total_examples=10
        )

    result = evaluate_compression_plans(
        model,
        {"first-low": plan},
        evaluators={"test": evaluator},
        metric_directions={"test": {"acc": "higher", "loss": "lower"}},
        decomposition_backends={"mpo": backend},
        execution_backends={"mpo": backend},
        collect_layer_metrics=True,
    )
    path = tmp_path / "report.md"
    text = format_sensitivity_report(result, path, task_name="test")
    assert (
        path.read_text() == text == format_sensitivity_report(result, task_name="test")
    )
    for expected in [
        "# Sensitivity Report",
        "## Baseline Metrics",
        "## Cases",
        "## Metrics",
        "## Layers",
        "mpo",
        "| first-low | auxiliary | 11 |  |",
        "- Total evaluation examples: 10",
    ]:
        assert expected in text
