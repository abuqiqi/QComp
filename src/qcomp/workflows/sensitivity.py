"""执行与张量网络表示、计算后端和具体评测任务解耦的敏感性分析。

本模块接收外部构造的压缩方案、backend 映射和模型 evaluator，先评测未压缩基线，
再按顺序临时安装每个方案、统计逐层与完整模型压缩指标、评测质量变化并恢复原始层。
它复用 compress workflow 和 evaluation 层，不创建 backend、不生成结构 rank，也不
读取数据集。

主要内容：
- ``MetricDirection``：限定敏感性指标的优化方向。
- ``ModelEvaluator``：描述接收当前模型并返回通用评测结果的函数。
- ``SensitivityCase``：使用名称和 ``CompressionPlan`` 描述一次压缩实验。
- ``SensitivityCaseResult``：保存单次实验的质量退化、压缩指标和耗时。
- ``SensitivityResult``：组合未压缩基线与全部有序实验结果。
- ``analyze_sensitivity``：顺序执行压缩、评测和恢复流程。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from torch import nn

from ..backends import TensorNetworkBackend
from ..evaluation import (
    CompressionMetrics,
    EvaluationResult,
    ModelCompressionMetrics,
    compression_metrics,
    model_compression_metrics,
)
from .compress import (
    CompressionPlan,
    compress_model,
    restore_compressed_model,
)

MetricDirection: TypeAlias = Literal["higher", "lower"]
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


@dataclass(frozen=True)
class SensitivityResult:
    """保存未压缩基线和全部敏感性实验结果。"""

    baseline_evaluation: EvaluationResult
    baseline_evaluation_seconds: float
    case_results: tuple[SensitivityCaseResult, ...]


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
) -> SensitivityResult:
    """顺序评测多个压缩方案相对于未压缩模型的指标退化。

    参数：
        model: 包含各压缩目标的完整 PyTorch 模型。
        cases: 按执行顺序排列的敏感性实验。
        decomposition_backends: 按表示名称提供分解后端的映射。
        execution_backends: 按表示名称提供执行后端的映射。
        evaluator: 接收当前模型并返回通用评测结果的函数。
        metric_directions: 待比较指标及其 ``higher`` 或 ``lower`` 方向。

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
            case_results.append(
                SensitivityCaseResult(
                    case=case,
                    evaluation=evaluation,
                    metric_degradations=degradations,
                    layer_compressions=layer_compressions,
                    model_compression=model_compression,
                    compression_seconds=compression_seconds,
                    evaluation_seconds=evaluation_seconds,
                )
            )
        finally:
            if compression_result is not None:
                restore_compressed_model(model, compression_result)

    return SensitivityResult(
        baseline_evaluation=baseline_evaluation,
        baseline_evaluation_seconds=baseline_seconds,
        case_results=tuple(case_results),
    )
