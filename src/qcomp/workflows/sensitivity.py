"""组织单矩阵敏感度实验，复用通用方案评测并输出既有日志与报告格式。

本模块负责资源组装、目标筛选和报告转换，实际压缩评测由 evaluate workflow 执行。
主要内容：
- ``SensitivityExperimentConfig``、``SensitivityExperimentResult``：实验配置和报告定位。
- ``run_sensitivity_experiment``：生成单矩阵计划并调用通用多方案评测。
- ``sensitivity_case_record``、``format_sensitivity_report``：将通用结果转换为敏感度产物。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn

from ..backends import get_backend
from ..evaluation import LMEvalConfig, LMEvalEvaluator, resolve_metric_directions
from ..model import ModelLoadConfig, list_linears, load_causal_lm
from ..runtime import load_runtime_config
from ..logging import log_event
from ..storage import ArtifactPaths
from .compress import CompressionPlan, CompressionTarget
from .evaluate import (
    CompressionEvaluationResult,
    CompressionPlanEvaluation,
    evaluate_compression_plans,
)


def sensitivity_case_record(
    result: CompressionPlanEvaluation, task_name: str
) -> dict[str, Any]:
    """将单次敏感性结果整理成独立的实验记录字段。

    参数：
        result: 当前实验的质量指标、退化量、压缩指标和耗时。
        task_name: 通用结果中的任务键。

    返回：
        包含实验名称、指标和耗时的字典，可交给日志或其他输出接口。
        记录不包含模型、张量、文件路径或写入时间。
    """

    return {
        "case": result.name,
        "metrics": dict(result.evaluations[task_name].evaluation.metrics),
        "degradations": dict(result.metric_degradations[task_name]),
        "layers": {
            module_path: {
                "compression_ratio": value.compression_ratio,
                "relative_error": value.relative_error,
            }
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


@dataclass(frozen=True)
class SensitivityExperimentConfig:
    """配置逐层 lm-eval 敏感性实验的模型、评测、范围和输出。"""

    name: str
    evaluation: LMEvalConfig
    metrics: tuple[str, ...]
    model: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    runtime_config: str | Path | None = None
    decomposition_provider: str = "tensorly"
    execution_provider: str = "tensorly"
    decomposition_dtype: torch.dtype = torch.float32
    start_layer_index: int = 0
    max_layers: int | None = None
    artifact_root: str | Path = "artifacts"
    output: str | Path | None = None
    log: str | Path | None = None

    def __post_init__(self) -> None:
        """校验实验名称、层范围和分解类型，无效时抛出 ValueError。"""

        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.start_layer_index < 0:
            raise ValueError("start_layer_index must be non-negative")
        if self.max_layers is not None and self.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if not self.decomposition_dtype.is_floating_point:
            raise ValueError("decomposition_dtype must be floating-point")


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
    """加载模型，为选中层分别创建 case，执行 lm-eval 分析并保存日志和报告。

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
    directions = resolve_metric_directions(config.metrics)
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
    representations = dict.fromkeys(
        plan.targets[0].representation for plan in plans.values()
    )
    decomposition_backends = {
        representation: get_backend(config.decomposition_provider, representation)
        for representation in representations
    }
    execution_backends = {
        representation: get_backend(config.execution_provider, representation)
        for representation in representations
    }
    evaluator = LMEvalEvaluator(
        resources.tokenizer,
        config.evaluation,
        runtime_config_path=config.runtime_config,
    )
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
    log_path = Path(config.log) if config.log else output_path.with_suffix(".jsonl")
    run_id = uuid4().hex
    log_event(
        log_path,
        "experiment_started",
        run_id=run_id,
        name=config.name,
        task=config.evaluation.task,
        metrics=tuple(directions),
        metric_directions=directions,
        layers=len(plans),
        first=selected[0][0],
        last=selected[-1][0],
        model_name_or_path=model_config.model_name_or_path,
        device=model_config.device,
        model_dtype=str(model_config.dtype),
        decomposition_dtype=str(config.decomposition_dtype),
        decomposition_provider=config.decomposition_provider,
        execution_provider=config.execution_provider,
        seed=config.evaluation.seed,
        num_fewshot=config.evaluation.num_fewshot,
        batch_size=config.evaluation.batch_size,
        max_length=config.evaluation.max_length,
        limit=config.evaluation.limit,
        sample_start_index=config.evaluation.sample_start_index,
        apply_chat_template=config.evaluation.apply_chat_template,
        trust_remote_code=model_config.trust_remote_code,
        offline=runtime.offline,
        start_layer_index=config.start_layer_index,
        max_layers=config.max_layers,
    )

    def on_plan_result(result: CompressionPlanEvaluation) -> None:
        """将结果参数 result 转换为事件，关联当前运行并追加到日志。"""

        log_event(
            log_path,
            "case_completed",
            run_id=run_id,
            **sensitivity_case_record(result, config.evaluation.task),
        )

    result = evaluate_compression_plans(
        resources.model,
        plans,
        decomposition_backends=decomposition_backends,
        execution_backends=execution_backends,
        decomposition_dtype=config.decomposition_dtype,
        evaluators={config.evaluation.task: evaluator},
        metric_directions={config.evaluation.task: directions},
        collect_layer_metrics=True,
        on_plan_result=on_plan_result,
    )
    format_sensitivity_report(result, output_path, task_name=config.evaluation.task)
    log_event(log_path, "experiment_completed", run_id=run_id, report=str(output_path))
    return SensitivityExperimentResult(result, output_path)
