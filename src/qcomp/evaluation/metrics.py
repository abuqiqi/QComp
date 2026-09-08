"""统一定义敏感性分析常用评测指标的优化方向。

本模块按规范化指标名称保存 ``higher`` 或 ``lower`` 语义，并把调用方选择的指标序列
转换成 sensitivity workflow 已有的方向映射。这里只描述指标本身，不维护 lm-eval task
与主指标之间的对应关系。

主要内容：
- ``MetricDirection``：限定指标优化方向。
- ``metric_direction``：查询一个已注册指标的优化方向。
- ``resolve_metric_directions``：批量构造 sensitivity 使用的指标方向映射。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

MetricDirection: TypeAlias = Literal["higher", "lower"]

_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "acc": "higher",
    "acc_norm": "higher",
    "exact_match": "higher",
    "exact_match_remove_whitespace": "higher",
    "exact_match_strict_match": "higher",
    "exact_match_flexible_extract": "higher",
    "f1": "higher",
    "loss": "lower",
    "perplexity": "lower",
    "word_perplexity": "lower",
    "byte_perplexity": "lower",
    "bits_per_byte": "lower",
}


def _normalize_metric_name(metric: str) -> str:
    """规范并校验一个指标名称。

    参数：
        metric: 待查询或批量解析的指标名称。

    返回：
        去除首尾空白并转为小写的指标名称。

    异常：
        ValueError: 指标名称为空时抛出。
    """

    normalized = metric.strip().lower()
    if not normalized:
        raise ValueError("metric name must not be empty")
    return normalized


def metric_direction(metric: str) -> MetricDirection:
    """返回一个常用指标的优化方向。

    参数：
        metric: lm-eval 转换后的指标名称。

    返回：
        ``higher`` 或 ``lower``。

    异常：
        ValueError: 指标尚未在公共 registry 中注册时抛出。
    """

    normalized = _normalize_metric_name(metric)
    try:
        return _METRIC_DIRECTIONS[normalized]
    except KeyError as error:
        raise ValueError(
            f"metric direction is not registered for {normalized!r}"
        ) from error


def resolve_metric_directions(
    metrics: Sequence[str],
) -> dict[str, MetricDirection]:
    """把指标名称序列转换成 sensitivity 使用的方向映射。

    参数：
        metrics: 一个或多个已注册指标名称。

    返回：
        保持输入顺序的规范指标名称到优化方向映射。

    异常：
        TypeError: 把单个字符串误传为指标序列时抛出。
        ValueError: 指标序列为空、包含重复名称或未知指标时抛出。
    """

    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names")
    normalized = tuple(_normalize_metric_name(metric) for metric in metrics)
    if not normalized:
        raise ValueError("metrics must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("metrics must not contain duplicates")
    return {name: metric_direction(name) for name in normalized}
