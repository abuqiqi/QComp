"""在统一选层 JSON 和压缩计划之间转换，并在加载时检查实际模型。

只解析执行结构，不读取分析来源、不运行分解或修改模型；返回的计划交给压缩 workflow。
主要内容：
- ``compression_plan_to_dict``、``compression_plan_from_dict``：读写完整逐目标 spec。
- ``load_compression_plan``：读取文件并自动验证目标 Linear 与矩阵维度。
"""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from torch import nn

from ..model import find_linear
from ..representations import MPOSpec
from .compress import CompressionPlan, CompressionTarget


def _fields(value: Any, expected: set[str], context: str) -> None:
    """校验执行对象的精确字段集合，拒绝遗漏及拼错的字段。"""
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{context}: 字段必须为 {sorted(expected)}")


def compression_plan_from_dict(data: Mapping[str, Any]) -> CompressionPlan:
    """将 compression_plan 对象解析为计划；仅支持完整 MPO spec，不读取模型。"""
    _fields(data, {"targets"}, "compression_plan")
    if not isinstance(data["targets"], list):
        raise ValueError("compression_plan.targets 必须为数组")
    targets = []
    for index, entry in enumerate(data["targets"]):
        context = f"target[{index}]"
        _fields(entry, {"module_path", "representation", "spec"}, context)
        path = entry["module_path"]
        if not isinstance(path, str) or not path or path != path.strip():
            raise ValueError(f"{context}: module_path 无效")
        context = path
        if entry["representation"] != "mpo":
            raise ValueError(
                f"{context}: 不支持 representation={entry['representation']!r}"
            )
        spec = entry["spec"]
        _fields(spec, {"out_modes", "in_modes", "ranks"}, context)
        for name, values in spec.items():
            if (
                not isinstance(values, list)
                or not values
                or any(type(v) is not int or v <= 0 for v in values)
            ):
                raise ValueError(f"{context}: {name} 必须为正整数数组")
        try:
            parsed = MPOSpec(**spec)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{context}: {error}") from error
        targets.append(CompressionTarget(path, "mpo", parsed))
    return CompressionPlan(tuple(targets))


def compression_plan_to_dict(plan: CompressionPlan) -> dict[str, Any]:
    """把 plan 转为可 JSON 序列化的执行对象，保留目标顺序及独立 ranks。"""
    targets = []
    for target in plan.targets:
        if target.representation != "mpo" or not isinstance(target.spec, MPOSpec):
            raise ValueError(f"{target.module_path}: 当前仅支持 MPO spec 序列化")
        targets.append(
            dict(
                module_path=target.module_path,
                representation=target.representation,
                spec={
                    name: list(getattr(target.spec, name))
                    for name in ("out_modes", "in_modes", "ranks")
                },
            )
        )
    result = {"targets": targets}
    compression_plan_from_dict(result)
    return result


def load_compression_plan(path: str | Path, *, model: nn.Module) -> CompressionPlan:
    """读取完整选层 JSON 并自动核对 model 的目标；失败抛出 ValueError，不修改模型。

    参数：
        path: 最新格式的选层文件。
        model: 调用方已加载的模型，所有目标须为无 bias Linear 且维度与 spec 一致。

    返回：
        可直接交给 compress_model 的 CompressionPlan。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("kind") != "qcomp_compression_selection"
        or "compression_plan" not in data
    ):
        raise ValueError("文件格式不符合当前要求，请使用最新版页面重新导出")
    plan = compression_plan_from_dict(data["compression_plan"])
    for target in plan.targets:
        try:
            linear = find_linear(model, target.module_path)
            shape = (linear.out_features, linear.in_features)
            expected = (target.spec.out_features, target.spec.in_features)
            if shape != expected:
                raise ValueError(f"维度不匹配：模型 {shape}，计划 {expected}")
        except (AttributeError, ValueError, TypeError) as error:
            raise ValueError(f"{target.module_path}: {error}") from error
    return plan
