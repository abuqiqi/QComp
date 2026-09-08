"""定义与具体数据集和指标实现无关的评测任务与结果。

本模块使用稳定名称描述数据集、split、预处理配置和请求的指标名称，并用通用映射保存
评测输出。具体 evaluator 负责解释任务字段、读取评测数据并执行相应的 prompt、batching
和指标规则；训练用 data 层与此处相互独立。

主要内容：
- ``EvaluationTask``：描述一次可复现评测使用的数据和请求指标。
- ``EvaluationResult``：保存任务、动态指标、样本数和可选 token 数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationTask:
    """描述数据集、预处理方式和需要计算的指标。"""

    name: str
    dataset: str
    split: str
    preprocessing: str
    requested_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        """规范任务字段和请求的指标名称，并确认内容完整且不重复。

        异常：
            ValueError: 必填名称为空、请求指标为空或指标名称重复时抛出。
        """

        for field_name in ("name", "dataset", "split", "preprocessing"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        requested_metrics = tuple(
            metric.strip().lower() for metric in self.requested_metrics
        )
        if not requested_metrics or any(not metric for metric in requested_metrics):
            raise ValueError("requested_metrics must contain non-empty names")
        if len(set(requested_metrics)) != len(requested_metrics):
            raise ValueError("requested_metrics must not contain duplicates")
        object.__setattr__(self, "requested_metrics", requested_metrics)


@dataclass(frozen=True)
class EvaluationResult:
    """保存一个评测任务产生的通用指标和统计范围。"""

    task: EvaluationTask
    metrics: Mapping[str, float]
    evaluated_examples: int
    evaluated_tokens: int | None = None
    total_examples: int | None = None
