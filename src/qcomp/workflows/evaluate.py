"""评测多个压缩方案在多个任务上的效果，复用压缩操作并保证恢复原模型。

调用方提供已加载模型、具名计划、evaluator 和 backend；本模块只编排、计时和统计，
不加载资源或写文件。回调允许调用方及时保存产物，返回结果不保留压缩张量。
主要内容：
- ``TimedEvaluation``、``CompressionPlanEvaluation``、``CompressionEvaluationResult``：通用结果。
- ``ModelEvaluator``：接收模型并返回 EvaluationResult 的可调用接口。
- ``evaluate_compression_plans``：共享 baseline，逐方案联合压缩并评测全部任务。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
import time
from typing import Any, TypeAlias

import torch
from torch import nn

from ..backends import TensorNetworkBackend
from ..evaluation import (
    CompressionMetrics,
    EvaluationResult,
    MetricDirection,
    ModelCompressionMetrics,
    compression_metrics,
    model_compression_metrics,
)
from ..representations import TensorNetworkArtifact
from .compress import CompressionPlan, compress_model, restore_compressed_model

ModelEvaluator: TypeAlias = Callable[[nn.Module], EvaluationResult]


@dataclass(frozen=True)
class TimedEvaluation:
    """保存一个任务的通用评测结果和同步后的耗时（秒）。"""

    evaluation: EvaluationResult
    seconds: float


@dataclass(frozen=True)
class CompressionPlanEvaluation:
    """保存单个具名计划的多任务结果与数值统计，不持有模型或 artifact。"""

    name: str
    plan: CompressionPlan
    evaluations: Mapping[str, TimedEvaluation]
    metric_degradations: Mapping[str, Mapping[str, float]]
    model_compression: ModelCompressionMetrics
    layer_compressions: Mapping[str, CompressionMetrics]
    compression_seconds: float


@dataclass(frozen=True)
class CompressionEvaluationResult:
    """组合各任务共同 baseline 与按输入顺序保存的方案结果。"""

    baseline: Mapping[str, TimedEvaluation]
    plan_results: tuple[CompressionPlanEvaluation, ...]


def _synchronize(model: nn.Module) -> None:
    """计时前后同步模型所在 CUDA 设备；没有参数或使用 CPU 时不操作。"""
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)


def _directions(values: Mapping[str, MetricDirection]) -> dict[str, MetricDirection]:
    """规范单个任务的指标名称，拒绝空映射、重名和未知方向。"""
    if not values:
        raise ValueError("metric_directions must not be empty for an evaluator")
    result = {}
    for metric, direction in values.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric name must not be empty")
        name = metric.strip().lower()
        if name in result:
            raise ValueError(f"duplicate normalized metric name: {name!r}")
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"metric direction for {name!r} must be 'higher' or 'lower'"
            )
        result[name] = direction
    return result


def _evaluate(
    model: nn.Module,
    evaluator: ModelEvaluator,
    directions: Mapping[str, MetricDirection],
    baseline: TimedEvaluation | None,
) -> TimedEvaluation:
    """执行并校验一个任务，复制数值指标以免 evaluator 后续修改共享结果。"""
    model.eval()
    _synchronize(model)
    started = time.perf_counter()
    evaluation = evaluator(model)
    _synchronize(model)
    seconds = time.perf_counter() - started
    for metric in directions:
        if metric not in evaluation.metrics:
            raise ValueError(f"evaluation result does not contain metric {metric!r}")
        if not math.isfinite(float(evaluation.metrics[metric])):
            raise ValueError(f"evaluation metric {metric!r} must be finite")
    if (
        type(evaluation.evaluated_examples) is not int
        or evaluation.evaluated_examples <= 0
    ):
        raise ValueError("evaluated_examples must be a positive integer")
    if evaluation.total_examples is not None and (
        type(evaluation.total_examples) is not int
        or evaluation.total_examples < evaluation.evaluated_examples
    ):
        raise ValueError("total_examples must cover evaluated_examples")
    if baseline is not None:
        previous = baseline.evaluation
        if previous.task != evaluation.task:
            raise ValueError("evaluations must use the same EvaluationTask as baseline")
        if (
            previous.evaluated_examples != evaluation.evaluated_examples
            or previous.total_examples != evaluation.total_examples
        ):
            raise ValueError("evaluation sample counts must match baseline")
    return TimedEvaluation(
        replace(
            evaluation,
            metrics={name: float(value) for name, value in evaluation.metrics.items()},
        ),
        seconds,
    )


def evaluate_compression_plans(
    model: nn.Module,
    plans: Mapping[str, CompressionPlan],
    *,
    evaluators: Mapping[str, ModelEvaluator],
    metric_directions: Mapping[str, Mapping[str, MetricDirection]],
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    decomposition_dtype: torch.dtype | None = None,
    collect_layer_metrics: bool = False,
    on_evaluation: Callable[[str | None, str, TimedEvaluation], None] | None = None,
    on_compressed: (
        Callable[
            [str, Mapping[str, TensorNetworkArtifact], ModelCompressionMetrics, float],
            None,
        ]
        | None
    ) = None,
    on_plan_result: Callable[[CompressionPlanEvaluation], None] | None = None,
) -> CompressionEvaluationResult:
    """共享各任务 baseline，按 plans 顺序执行联合压缩及多指标评测。

    参数：
        model: 已加载模型；成功或失败均恢复原始 Linear 和各模块训练状态。
        plans: 非空的名称到 CompressionPlan 映射，名称和迭代顺序原样保留。
        evaluators: 任务键到评测函数的映射；空映射表示仅压缩并统计。
        metric_directions: 每个任务关注的指标到 higher/lower 的映射。
        decomposition_backends: 各表示的分解后端。
        execution_backends: 各表示的执行后端。
        decomposition_dtype: 分解精度；结果由压缩接口转回原层精度。
        collect_layer_metrics: 是否额外重建逐矩阵权重以统计误差，不增加任务评测。
        on_evaluation: 一项评测通过校验后通知；baseline 的方案名为 None。
        on_compressed: 联合压缩及整模统计完成后通知，早于逐层统计和任务评测。
            artifact 仅供同步保存，回调不应跨方案保留引用。
        on_plan_result: 该方案完成并恢复模型后通知。

    返回：
        数值与结构结果，不保存模型、artifact 或文件路径。

    异常：
        ValueError: 配置、指标、任务或样本范围不一致。
        Exception: 分解、评测、统计或回调失败即停止，恢复后向调用方传播。
    """
    if not isinstance(plans, Mapping) or not plans:
        raise ValueError("plans must be a non-empty mapping")
    for name in (*plans, *evaluators):
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                "plan and evaluator names must be non-empty without surrounding whitespace"
            )
    if set(evaluators) != set(metric_directions):
        raise ValueError("metric_directions keys must match evaluators")
    directions = {name: _directions(metric_directions[name]) for name in evaluators}
    if decomposition_dtype is not None and not decomposition_dtype.is_floating_point:
        raise ValueError("decomposition_dtype must be floating-point")
    for plan in plans.values():
        if not isinstance(plan, CompressionPlan):
            raise ValueError("plans must contain CompressionPlan values")
        for target in plan.targets:
            for role, backends in (
                ("decomposition", decomposition_backends),
                ("execution", execution_backends),
            ):
                backend = backends.get(target.representation)
                if backend is None or backend.representation != target.representation:
                    raise ValueError(
                        f"invalid {role} backend for {target.representation!r}"
                    )
    training = [(module, module.training) for module in model.modules()]
    baseline, results = {}, []
    try:
        for task, evaluator in evaluators.items():
            baseline[task] = _evaluate(model, evaluator, directions[task], None)
            if on_evaluation is not None:
                on_evaluation(None, task, baseline[task])
        for name, plan in plans.items():
            compression = None
            try:
                model.eval()
                _synchronize(model)
                started = time.perf_counter()
                compression = compress_model(
                    model,
                    plan,
                    decomposition_backends=decomposition_backends,
                    execution_backends=execution_backends,
                    trainable=False,
                    decomposition_dtype=decomposition_dtype,
                )
                _synchronize(model)
                seconds = time.perf_counter() - started
                model_metrics = model_compression_metrics(model, compression)
                if on_compressed is not None:
                    on_compressed(
                        name,
                        {
                            target.module_path: layer.tn_artifact
                            for target, layer in zip(
                                plan.targets, compression.layer_results, strict=True
                            )
                        },
                        model_metrics,
                        seconds,
                    )
                layer_metrics = {}
                if collect_layer_metrics:
                    layer_metrics = {
                        target.module_path: compression_metrics(
                            layer.replacement.original.weight.detach(),
                            layer.tn_artifact,
                        )
                        for target, layer in zip(
                            plan.targets, compression.layer_results, strict=True
                        )
                    }
                evaluations, degradations = {}, {}
                for task, evaluator in evaluators.items():
                    timed = _evaluate(
                        model, evaluator, directions[task], baseline[task]
                    )
                    evaluations[task] = timed
                    before, after = (
                        baseline[task].evaluation.metrics,
                        timed.evaluation.metrics,
                    )
                    degradations[task] = {
                        metric: (
                            before[metric] - after[metric]
                            if direction == "higher"
                            else after[metric] - before[metric]
                        )
                        for metric, direction in directions[task].items()
                    }
                    if on_evaluation is not None:
                        on_evaluation(name, task, timed)
                result = CompressionPlanEvaluation(
                    name,
                    plan,
                    evaluations,
                    degradations,
                    model_metrics,
                    layer_metrics,
                    seconds,
                )
            finally:
                if compression is not None:
                    restore_compressed_model(model, compression)
                    compression = None
                for module, was_training in training:
                    module.training = was_training
            results.append(result)
            if on_plan_result is not None:
                on_plan_result(result)
    finally:
        for module, was_training in training:
            module.training = was_training
    return CompressionEvaluationResult(baseline, tuple(results))
