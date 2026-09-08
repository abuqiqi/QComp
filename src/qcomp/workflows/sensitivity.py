"""执行与张量网络表示、计算后端和具体评测任务解耦的敏感性分析。

本模块接收外部构造的压缩方案、backend 映射和模型 evaluator，先评测未压缩基线，
再按顺序临时安装每个方案、统计逐层与完整模型压缩指标、评测质量变化并恢复原始层。
底层分析复用 compress workflow 和 evaluation 层；上层实验入口加载模型、创建
backend 和 lm-eval evaluator，并统一输出日志和报告。结构配置由调用方提供。

主要内容：
- ``SensitivityExperimentConfig``、``run_sensitivity_experiment``：配置并执行逐层实验。
- ``ModelEvaluator``：描述接收当前模型并返回通用评测结果的函数。
- ``SensitivityCase``：使用名称和 ``CompressionPlan`` 描述一次压缩实验。
- ``SensitivityCaseResult``：保存单次实验的质量退化、压缩指标和耗时。
- ``sensitivity_case_record``：将单次结果整理为实验记录字段。
- ``SensitivityResult``：组合未压缩基线与全部有序实验结果。
- ``analyze_sensitivity``：顺序执行压缩、评测和恢复流程。
- ``format_sensitivity_report``：生成 Markdown 报告并按需写入指定路径。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeAlias
from uuid import uuid4

import torch
from torch import nn

from ..backends import TensorNetworkBackend, get_backend
from ..evaluation import (
    LMEvalConfig,
    LMEvalEvaluator,
    resolve_metric_directions,
    CompressionMetrics,
    EvaluationResult,
    MetricDirection,
    ModelCompressionMetrics,
    compression_metrics,
    model_compression_metrics,
)
from ..model import ModelLoadConfig, list_linears, load_causal_lm
from ..runtime import load_runtime_config
from ..logging import log_event
from ..storage import ArtifactPaths
from .compress import (
    CompressionPlan,
    CompressionTarget,
    compress_model,
    restore_compressed_model,
)

ModelEvaluator: TypeAlias = Callable[[nn.Module], EvaluationResult]


@dataclass(frozen=True)
class SensitivityCase:
    """描述一次可以压缩一个或多个模型层的敏感性实验。"""

    name: str
    compression_plan: CompressionPlan

    def __post_init__(self) -> None:
        """确认实验名称包含有效内容。

        异常：
            ValueError: 实验名称为空时抛出。
        """

        name = self.name.strip()
        if not name:
            raise ValueError("sensitivity case name must not be empty")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class SensitivityCaseResult:
    """保存一次压缩实验的质量、规模和执行时间结果。"""

    case: SensitivityCase
    evaluation: EvaluationResult
    metric_degradations: Mapping[str, float]
    layer_compressions: Mapping[str, CompressionMetrics]
    model_compression: ModelCompressionMetrics
    compression_seconds: float
    evaluation_seconds: float


def sensitivity_case_record(result: SensitivityCaseResult) -> dict[str, Any]:
    """将单次敏感性结果整理成独立的实验记录字段。

    参数：
        result: 当前实验的质量指标、退化量、压缩指标和耗时。

    返回：
        包含实验名称、指标和耗时的字典，可交给日志或其他输出接口。
        记录不包含模型、张量、文件路径或写入时间。
    """

    return {
        "case": result.case.name,
        "metrics": dict(result.evaluation.metrics),
        "degradations": dict(result.metric_degradations),
        "layers": {
            module_path: {
                "compression_ratio": value.compression_ratio,
                "relative_error": value.relative_error,
            }
            for module_path, value in result.layer_compressions.items()
        },
        "model_ratio": result.model_compression.compression_ratio,
        "compression_seconds": result.compression_seconds,
        "evaluation_seconds": result.evaluation_seconds,
    }


@dataclass(frozen=True)
class SensitivityResult:
    """保存未压缩基线、全部实验结果及可选的已落盘报告路径。"""

    baseline_evaluation: EvaluationResult
    baseline_evaluation_seconds: float
    case_results: tuple[SensitivityCaseResult, ...]
    report_path: Path | None = None


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


def _synchronize_model_device(model: nn.Module) -> None:
    """等待模型所在 CUDA 设备上的操作完成。

    参数：
        model: 用于确定当前执行设备的 PyTorch 模型。
    """

    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)


def _timed_evaluation(
    model: nn.Module,
    evaluator: ModelEvaluator,
) -> tuple[EvaluationResult, float]:
    """执行一次模型评测并返回同步后的耗时。

    参数：
        model: 当前待评测的模型。
        evaluator: 接收模型并返回通用评测结果的函数。

    返回：
        评测结果和以秒为单位的执行时间。
    """

    _synchronize_model_device(model)
    started = time.perf_counter()
    evaluation = evaluator(model)
    _synchronize_model_device(model)
    return evaluation, time.perf_counter() - started


def _normalize_metric_directions(
    metric_directions: Mapping[str, MetricDirection],
) -> dict[str, MetricDirection]:
    """规范并验证需要分析的指标名称和优化方向。

    参数：
        metric_directions: 指标名称到 ``higher`` 或 ``lower`` 的映射。

    返回：
        名称已去除空白并转为小写的指标方向映射。

    异常：
        ValueError: 映射为空、指标名称为空、名称重复或方向无效时抛出。
    """

    if not metric_directions:
        raise ValueError("metric_directions must not be empty")
    normalized: dict[str, MetricDirection] = {}
    for metric, direction in metric_directions.items():
        name = metric.strip().lower()
        if not name:
            raise ValueError("metric name must not be empty")
        if name in normalized:
            raise ValueError(f"duplicate normalized metric name: {name!r}")
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"metric direction for {name!r} must be 'higher' or 'lower'"
            )
        normalized[name] = direction
    return normalized


def _metric_values(
    evaluation: EvaluationResult,
    metric_names: Sequence[str],
) -> dict[str, float]:
    """从通用评测结果中读取指定指标。

    参数：
        evaluation: evaluator 返回的通用评测结果。
        metric_names: 需要读取的规范化指标名称。

    返回：
        指标名称到浮点数值的映射。

    异常：
        ValueError: 评测结果缺少任一指定指标时抛出。
    """

    values: dict[str, float] = {}
    for name in metric_names:
        if name not in evaluation.metrics:
            raise ValueError(f"evaluation result does not contain metric {name!r}")
        values[name] = float(evaluation.metrics[name])
    return values


def analyze_sensitivity(
    model: nn.Module,
    cases: Sequence[SensitivityCase],
    *,
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    evaluator: ModelEvaluator,
    metric_directions: Mapping[str, MetricDirection],
    on_case_result: Callable[[SensitivityCaseResult], None] | None = None,
    decomposition_dtype: torch.dtype | None = None,
) -> SensitivityResult:
    """顺序评测多个压缩方案相对于未压缩模型的指标退化。

    参数：
        model: 包含各压缩目标的完整 PyTorch 模型。
        cases: 按执行顺序排列的敏感性实验。
        decomposition_backends: 按表示名称提供分解后端的映射。
        execution_backends: 按表示名称提供执行后端的映射。
        evaluator: 接收当前模型并返回通用评测结果的函数。
        metric_directions: 待比较指标及其 ``higher`` 或 ``lower`` 方向。
        on_case_result: 每个 case 恢复原模型后接收结果的可选回调。
        decomposition_dtype: 可选分解浮点类型；省略时使用原权重类型。

    返回：
        未压缩基线、基线耗时和全部有序实验结果。

    异常：
        ValueError: 实验为空、名称重复、指标配置无效或评测任务不一致时抛出。
        Exception: 压缩、指标统计或评测失败时，在恢复模型后继续抛出原始异常。
    """

    if not cases:
        raise ValueError("cases must not be empty")
    case_names = tuple(case.name for case in cases)
    if len(set(case_names)) != len(case_names):
        raise ValueError("sensitivity case names must be unique")
    directions = _normalize_metric_directions(metric_directions)

    baseline_evaluation, baseline_seconds = _timed_evaluation(model, evaluator)
    baseline_values = _metric_values(baseline_evaluation, tuple(directions))
    case_results: list[SensitivityCaseResult] = []

    for case in cases:
        compression_result = None
        try:
            _synchronize_model_device(model)
            compression_started = time.perf_counter()
            compression_result = compress_model(
                model,
                case.compression_plan,
                decomposition_backends=decomposition_backends,
                execution_backends=execution_backends,
                trainable=False,
                decomposition_dtype=decomposition_dtype,
            )
            _synchronize_model_device(model)
            compression_seconds = time.perf_counter() - compression_started

            layer_compressions = {
                target.module_path: compression_metrics(
                    layer_result.replacement.original.weight.detach(),
                    layer_result.tn_artifact,
                )
                for target, layer_result in zip(
                    case.compression_plan.targets,
                    compression_result.layer_results,
                    strict=True,
                )
            }
            model_compression = model_compression_metrics(model, compression_result)
            evaluation, evaluation_seconds = _timed_evaluation(model, evaluator)
            if evaluation.task != baseline_evaluation.task:
                raise ValueError(
                    "sensitivity evaluations must use the same EvaluationTask"
                )
            values = _metric_values(evaluation, tuple(directions))
            degradations = {
                name: (
                    baseline_values[name] - values[name]
                    if direction == "higher"
                    else values[name] - baseline_values[name]
                )
                for name, direction in directions.items()
            }
            case_result = SensitivityCaseResult(
                case=case,
                evaluation=evaluation,
                metric_degradations=degradations,
                layer_compressions=layer_compressions,
                model_compression=model_compression,
                compression_seconds=compression_seconds,
                evaluation_seconds=evaluation_seconds,
            )
        finally:
            if compression_result is not None:
                restore_compressed_model(model, compression_result)
        case_results.append(case_result)
        if on_case_result is not None:
            on_case_result(case_result)

    return SensitivityResult(
        baseline_evaluation=baseline_evaluation,
        baseline_evaluation_seconds=baseline_seconds,
        case_results=tuple(case_results),
    )


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
    result: SensitivityResult,
    output_path: str | Path | None = None,
) -> str:
    """将敏感性分析结果格式化为 Markdown，并按需写入磁盘。

    参数：
        result: ``analyze_sensitivity`` 返回的完整结果。
        output_path: 可选的 Markdown 输出文件路径。

    返回：
        与落盘内容相同的 Markdown 字符串。
    """

    task = result.baseline_evaluation.task
    lines = [
        "# Sensitivity Report",
        "",
        f"- Task: `{_markdown_cell(task.name)}`",
        f"- Dataset: `{_markdown_cell(task.dataset)}`",
        f"- Split: `{_markdown_cell(task.split)}`",
        f"- Preprocessing: `{_markdown_cell(task.preprocessing)}`",
        f"- Evaluated examples: {result.baseline_evaluation.evaluated_examples}",
        (
            "- Baseline evaluation seconds: "
            f"{_format_number(result.baseline_evaluation_seconds)}"
        ),
        "",
        "## Baseline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {_markdown_cell(name)} | {_format_number(float(value))} |"
        for name, value in result.baseline_evaluation.metrics.items()
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
    for case_result in result.case_results:
        lines.append(
            "| "
            f"{_markdown_cell(case_result.case.name)} | "
            f"{len(case_result.case.compression_plan.targets)} | "
            f"{_format_number(case_result.model_compression.compression_ratio)} | "
            f"{_format_number(case_result.model_compression.tensor_size_compression_ratio)} | "
            f"{_format_number(case_result.compression_seconds)} | "
            f"{_format_number(case_result.evaluation_seconds)} |"
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
    for case_result in result.case_results:
        for name, value in case_result.evaluation.metrics.items():
            degradation = case_result.metric_degradations.get(name)
            degradation_text = (
                _format_number(degradation) if degradation is not None else ""
            )
            lines.append(
                "| "
                f"{_markdown_cell(case_result.case.name)} | "
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
    for case_result in result.case_results:
        representations = {
            target.module_path: target.representation
            for target in case_result.case.compression_plan.targets
        }
        for module_path, metrics in case_result.layer_compressions.items():
            lines.append(
                "| "
                f"{_markdown_cell(case_result.case.name)} | "
                f"{_markdown_cell(module_path)} | "
                f"{_markdown_cell(representations[module_path])} | "
                f"{_format_number(metrics.compression_ratio)} | "
                f"{_format_number(metrics.tensor_size_compression_ratio)} | "
                f"{_format_number(metrics.relative_error)} |"
            )
    if result.baseline_evaluation.total_examples is not None:
        lines.insert(8, f"- Total evaluation examples: {result.baseline_evaluation.total_examples}")
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
) -> SensitivityResult:
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
        None if config.max_layers is None
        else config.start_layer_index + config.max_layers
    )
    selected = selected[config.start_layer_index : stop]
    if not selected:
        raise ValueError("the selected Linear range is empty")
    cases = []
    for path, linear in selected:
        target = make_target(path, linear)
        if target.module_path != path:
            raise ValueError("make_target must preserve the selected module path")
        cases.append(SensitivityCase(path, CompressionPlan(targets=(target,))))
    representations = dict.fromkeys(
        case.compression_plan.targets[0].representation for case in cases
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
            ArtifactPaths(config.artifact_root).root / "sensitivity"
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
        layers=len(cases),
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

    def on_case_result(result: SensitivityCaseResult) -> None:
        """将结果参数 result 转换为事件，关联当前运行并追加到日志。"""

        log_event(
            log_path, "case_completed", run_id=run_id, **sensitivity_case_record(result)
        )

    result = analyze_sensitivity(
        resources.model,
        tuple(cases),
        decomposition_backends=decomposition_backends,
        execution_backends=execution_backends,
        decomposition_dtype=config.decomposition_dtype,
        evaluator=evaluator,
        metric_directions=directions,
        on_case_result=on_case_result,
    )
    format_sensitivity_report(result, output_path)
    log_event(log_path, "experiment_completed", run_id=run_id, report=str(output_path))
    return replace(result, report_path=output_path)
