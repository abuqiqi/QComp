"""组织单矩阵敏感度实验，复用通用评测并原子保存完整结果与报告。

本模块负责资源组装、目标筛选和报告转换，实际压缩评测由 evaluate workflow 执行。
主要内容：
- ``SensitivityExperimentConfig``、``SensitivityExperimentResult``：实验配置和报告定位。
- ``run_sensitivity_experiment``：生成单矩阵计划并调用通用多方案评测。
- ``sensitivity_case_record``、``format_sensitivity_report``：将通用结果转换为敏感度产物。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone
from datetime import datetime as result_datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from torch import nn

from ..evaluation import (
    EvaluationTaskConfig,
    LMEvalEvaluator,
    evaluation_task_config_to_dict,
)
from ..model import ModelLoadConfig, list_linears, load_causal_lm
from ..runtime import load_runtime_config
from ..storage import ArtifactPaths
from .compress import CompressionExecutionConfig, CompressionPlan, CompressionTarget
from .compression_plan_io import compression_plan_to_dict
from .evaluate import (
    CompressionEvaluationResult,
    CompressionPlanEvaluation,
    evaluate_compression_plans,
)
from .sensitivity_io import model_config_digest, write_sensitivity_results


def sensitivity_case_record(
    result: CompressionPlanEvaluation, task_name: str
) -> dict[str, Any]:
    """将单次敏感性结果整理成独立的实验记录字段。

    参数：
        result: 当前实验的质量指标、退化量、压缩指标和耗时。
        task_name: 通用结果中的任务键。

    返回：
        包含实验名称、指标和耗时的字典，可交给标准结果或其他输出接口。
        记录包含完整计划和数值统计，不包含模型对象或张量。
    """

    return {
        "case": result.name,
        "compression_plan": compression_plan_to_dict(result.plan),
        "model_compression": asdict(result.model_compression),
        "metrics": dict(result.evaluations[task_name].evaluation.metrics),
        "degradations": dict(result.metric_degradations[task_name]),
        "layers": {
            module_path: asdict(value)
            for module_path, value in result.layer_compressions.items()
        },
        "model_ratio": result.model_compression.compression_ratio,
        "compression_seconds": result.compression_seconds,
        "evaluation_seconds": result.evaluations[task_name].seconds,
    }


@dataclass(frozen=True)
class SensitivityExperimentResult:
    """组合通用评测结果与已落盘的敏感度报告路径。"""

    evaluation: CompressionEvaluationResult
    report_path: Path
    results_path: Path


@dataclass(frozen=True)
class SensitivityExperimentConfig:
    """配置逐层 lm-eval 敏感性实验的模型、评测、范围和输出。"""

    name: str
    evaluation: EvaluationTaskConfig
    model: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    runtime_config: str | Path | None = None
    compression: CompressionExecutionConfig = field(
        default_factory=CompressionExecutionConfig
    )
    start_layer_index: int = 0
    max_layers: int | None = None
    artifact_root: str | Path = "artifacts"
    output: str | Path | None = None
    results: str | Path | None = None

    def __post_init__(self) -> None:
        """校验实验名称、层范围和分解类型，无效时抛出 ValueError。"""

        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.start_layer_index < 0:
            raise ValueError("start_layer_index must be non-negative")
        if self.max_layers is not None and self.max_layers <= 0:
            raise ValueError("max_layers must be positive")


def _format_number(value: float) -> str:
    """将报告中的浮点数格式化为紧凑且稳定的文本。

    参数：
        value: 待格式化的指标、比例或耗时。

    返回：
        最多包含六位有效数字的文本。
    """

    return f"{value:.6g}"


def _markdown_cell(value: object) -> str:
    """转义 Markdown 表格单元格中的分隔符和换行。

    参数：
        value: 待放入表格的值。

    返回：
        可以安全放入单行表格单元格的文本。
    """

    return str(value).replace("|", "\\|").replace("\n", " ")


def format_sensitivity_report(
    result: CompressionEvaluationResult,
    output_path: str | Path | None = None,
    *,
    task_name: str,
) -> str:
    """将敏感性分析结果格式化为 Markdown，并按需写入磁盘。

    参数：
        result: ``evaluate_compression_plans`` 返回的完整结果。
        output_path: 可选的 Markdown 输出文件路径。
        task_name: 要转换为单任务报告的 evaluator 键。

    返回：
        与落盘内容相同的 Markdown 字符串。
    """

    baseline = result.baseline[task_name]
    task = baseline.evaluation.task
    lines = [
        "# Sensitivity Report",
        "",
        f"- Task: `{_markdown_cell(task.name)}`",
        f"- Dataset: `{_markdown_cell(task.dataset)}`",
        f"- Split: `{_markdown_cell(task.split)}`",
        f"- Preprocessing: `{_markdown_cell(task.preprocessing)}`",
        f"- Evaluated examples: {baseline.evaluation.evaluated_examples}",
        ("- Baseline evaluation seconds: " f"{_format_number(baseline.seconds)}"),
        "",
        "## Baseline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {_markdown_cell(name)} | {_format_number(float(value))} |"
        for name, value in baseline.evaluation.metrics.items()
    )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            (
                "| Case | Targets | Model Compression Ratio | Tensor Size Ratio | "
                "Compression Seconds | Evaluation Seconds |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case_result in result.plan_results:
        lines.append(
            "| "
            f"{_markdown_cell(case_result.name)} | "
            f"{len(case_result.plan.targets)} | "
            f"{_format_number(case_result.model_compression.compression_ratio)} | "
            f"{_format_number(case_result.model_compression.tensor_size_compression_ratio)} | "
            f"{_format_number(case_result.compression_seconds)} | "
            f"{_format_number(case_result.evaluations[task_name].seconds)} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Case | Metric | Value | Degradation |",
            "|---|---|---:|---:|",
        ]
    )
    for case_result in result.plan_results:
        for name, value in case_result.evaluations[
            task_name
        ].evaluation.metrics.items():
            degradation = case_result.metric_degradations[task_name].get(name)
            degradation_text = (
                _format_number(degradation) if degradation is not None else ""
            )
            lines.append(
                "| "
                f"{_markdown_cell(case_result.name)} | "
                f"{_markdown_cell(name)} | "
                f"{_format_number(float(value))} | "
                f"{degradation_text} |"
            )
    lines.extend(
        [
            "",
            "## Layers",
            "",
            (
                "| Case | Module | Representation | Compression Ratio | "
                "Tensor Size Ratio | Relative Error |"
            ),
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for case_result in result.plan_results:
        representations = {
            target.module_path: target.representation
            for target in case_result.plan.targets
        }
        for module_path, metrics in case_result.layer_compressions.items():
            lines.append(
                "| "
                f"{_markdown_cell(case_result.name)} | "
                f"{_markdown_cell(module_path)} | "
                f"{_markdown_cell(representations[module_path])} | "
                f"{_format_number(metrics.compression_ratio)} | "
                f"{_format_number(metrics.tensor_size_compression_ratio)} | "
                f"{_format_number(metrics.relative_error)} |"
            )
    if baseline.evaluation.total_examples is not None:
        lines.insert(
            8, f"- Total evaluation examples: {baseline.evaluation.total_examples}"
        )
    report = "\n".join(lines) + "\n"
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    return report


def run_sensitivity_experiment(
    config: SensitivityExperimentConfig,
    *,
    select_linear: Callable[[str, nn.Linear], bool],
    make_target: Callable[[str, nn.Linear], CompressionTarget],
) -> SensitivityExperimentResult:
    """加载模型，为选中层分别创建 case，执行 lm-eval 分析并保存标准结果和报告。

    参数：
        config: 模型、评测、backend、层范围和输出配置。
        select_linear: 根据模块路径与 Linear 判断是否参与实验。
        make_target: 根据路径与 Linear 构造该层的压缩目标，路径必须保持一致。

    返回：
        包含基线和各层结果的敏感性分析结果。

    异常：
        ValueError: 指标、目标范围或回调返回的目标路径无效时抛出。
        Exception: 模型加载、分析或输出错误直接向调用方传播。
    """

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    evaluation = config.evaluation.evaluation
    directions = config.evaluation.metric_directions
    runtime = load_runtime_config(config.runtime_config)
    model_config = replace(
        config.model,
        model_name_or_path=config.model.model_name_or_path
        or runtime.model_name_or_path,
    )
    resources = load_causal_lm(model_config, runtime_config_path=config.runtime_config)
    selected = [
        (path, linear)
        for path, linear in list_linears(resources.model)
        if select_linear(path, linear)
    ]
    stop = (
        None
        if config.max_layers is None
        else config.start_layer_index + config.max_layers
    )
    selected = selected[config.start_layer_index : stop]
    if not selected:
        raise ValueError("the selected Linear range is empty")
    plans = {}
    for path, linear in selected:
        target = make_target(path, linear)
        if target.module_path != path:
            raise ValueError("make_target must preserve the selected module path")
        plans[path] = CompressionPlan(targets=(target,))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", config.name).strip("-._") or "experiment"
    last_index = config.start_layer_index + len(selected) - 1
    output_path = (
        Path(config.output)
        if config.output
        else (
            ArtifactPaths(config.artifact_root).root
            / "sensitivity"
            / slug
            / timestamp
            / f"layers-{config.start_layer_index:03d}-{last_index:03d}.md"
        )
    )
    results_path = (
        Path(config.results)
        if config.results
        else output_path.parent / "sensitivity_results.json"
    )
    if results_path.exists():
        raise FileExistsError(f"结果已存在：{results_path}")
    model_settings = getattr(resources.model, "config", None)
    model_data = model_settings.to_dict() if model_settings is not None else {}
    document = {
        "kind": "qcomp_sensitivity_results",
        "run_id": uuid4().hex,
        "name": config.name,
        "status": "running",
        "started_at": result_datetime.now(UTC).isoformat(),
        "finished_at": None,
        "expected_cases": list(plans),
        "model": {
            "name_or_path": model_config.model_name_or_path,
            "model_type": model_data.get("model_type"),
            "config": model_data,
            "config_sha256": model_config_digest(model_data),
            "dense_parameters": sum(p.numel() for p in resources.model.parameters()),
        },
        "execution": {
            "model_dtype": str(model_config.dtype),
            "decomposition_dtype": str(config.compression.decomposition_dtype),
            "decomposition_provider": config.compression.decomposition_provider,
            "execution_provider": config.compression.execution_provider,
            "trust_remote_code": model_config.trust_remote_code,
        },
        "evaluation_config": evaluation_task_config_to_dict(config.evaluation),
        "evaluation_task": None,
        "baseline": None,
        "cases": [],
        "provenance": {
            "kind": "native",
            "device": model_config.device,
            "offline": runtime.offline,
            "start_layer_index": config.start_layer_index,
            "max_layers": config.max_layers,
        },
        "report_path": str(output_path.resolve()),
    }
    write_sensitivity_results(results_path, document)

    def on_evaluation(name, task_name, timed) -> None:
        """基线 name 为 None 时持久化 task_name 对应的 timed 评测及真实任务信息。"""
        if name is None:
            document["evaluation_task"] = asdict(timed.evaluation.task)
            baseline_record = asdict(timed.evaluation)
            baseline_record.pop("task")
            document["baseline"] = {**baseline_record, "seconds": timed.seconds}
            write_sensitivity_results(results_path, document)

    def on_plan_result(result: CompressionPlanEvaluation) -> None:
        """追加已完成 result 的完整单层计划及统计，并原子更新进度。"""
        document["cases"].append(sensitivity_case_record(result, evaluation.task))
        write_sensitivity_results(results_path, document)

    try:
        decomposition_backends, execution_backends = config.compression.build_backends(
            plans
        )
        evaluator = LMEvalEvaluator(
            resources.tokenizer,
            evaluation,
            runtime_config_path=config.runtime_config,
        )
        result = evaluate_compression_plans(
            resources.model,
            plans,
            decomposition_backends=decomposition_backends,
            execution_backends=execution_backends,
            decomposition_dtype=config.compression.decomposition_dtype,
            evaluators={evaluation.task: evaluator},
            metric_directions={evaluation.task: directions},
            collect_layer_metrics=True,
            on_evaluation=on_evaluation,
            on_plan_result=on_plan_result,
        )
        document["status"] = "completed"
        document["finished_at"] = result_datetime.now(UTC).isoformat()
        write_sensitivity_results(results_path, document)
    except (Exception, KeyboardInterrupt) as error:
        document["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        document["finished_at"] = result_datetime.now(UTC).isoformat()
        document["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            write_sensitivity_results(results_path, document)
        except Exception as save_error:  # noqa: BLE001 - 保存失败不得掩盖原实验异常
            error.add_note(f"保存失败状态时出错：{save_error}")
        raise
    format_sensitivity_report(result, output_path, task_name=evaluation.task)
    return SensitivityExperimentResult(result, output_path, results_path)
