"""编排单个或多个 PyTorch Linear 的张量网络压缩。

本模块使用压缩目标与计划描述待处理的模型层。每个目标独立声明张量网络表示和结构
配置，批量执行时从外部 backend 映射选择对应实现，生成通用 artifact 并安装
``TensorNetworkLinear``。批量操作复用单层压缩，并在失败时恢复本次已经替换的层；
它复用 model 层的结构操作，不实现张量分解、张量收缩或评测逻辑。

主要内容：
- ``CompressionTarget``：保存一个目标层的模块路径、表示名称和结构配置。
- ``CompressionPlan``：保存可以采用不同表示的有序压缩目标。
- ``LinearCompressionResult``：保存压缩 artifact 和可用于恢复模型的替换记录。
- ``ModelCompressionResult``：保存模型级批量压缩产生的逐层结果。
- ``compress_linear``：完成单层分解、模型层构造和安装。
- ``compress_model``：按照计划压缩多个模型层，并保证失败时恢复模型。
- ``restore_compressed_model``：释放批量压缩层并恢复原始 Linear。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from ..backends import TensorNetworkBackend
from ..model import LinearReplacement, find_linear, replace_linear, restore_linear
from ..representations import TensorNetworkArtifact

SpecT = TypeVar("SpecT")


@dataclass(frozen=True)
class CompressionTarget(Generic[SpecT]):
    """描述一个待压缩模型层及其张量网络结构配置。"""

    module_path: str
    representation: str
    spec: SpecT

    def __post_init__(self) -> None:
        """规范表示名称，并确认目标模块路径和表示名称有效。

        异常：
            ValueError: 模块路径或表示名称为空时抛出。
        """

        if not self.module_path.strip():
            raise ValueError("module_path must not be empty")
        representation = self.representation.strip().lower()
        if not representation:
            raise ValueError("representation must not be empty")
        object.__setattr__(self, "representation", representation)


@dataclass(frozen=True)
class CompressionPlan:
    """描述可以混合多种张量网络表示的有序压缩目标。"""

    targets: tuple[CompressionTarget[Any], ...]

    def __post_init__(self) -> None:
        """确认计划非空且目标路径不重复。

        异常：
            ValueError: 目标列表为空或目标路径重复时抛出。
        """

        if not self.targets:
            raise ValueError("targets must not be empty")
        paths = tuple(target.module_path for target in self.targets)
        if len(set(paths)) != len(paths):
            raise ValueError("target module paths must be unique")


@dataclass(frozen=True)
class LinearCompressionResult:
    """保存单层压缩结果和原始模型层的恢复信息。"""

    tn_artifact: TensorNetworkArtifact
    replacement: LinearReplacement


@dataclass(frozen=True)
class ModelCompressionResult:
    """保存模型级批量压缩产生的有序逐层结果。"""

    layer_results: tuple[LinearCompressionResult, ...]


def compress_linear(
    model: nn.Module,
    target: str,
    spec: SpecT,
    *,
    decomposition_backend: TensorNetworkBackend[SpecT],
    execution_backend: TensorNetworkBackend[Any],
    trainable: bool,
    decomposition_dtype: torch.dtype | None = None,
) -> LinearCompressionResult:
    """分解目标 Linear，并安装执行后端构造的压缩模型层。

    参数：
        model: 包含目标 Linear 的 PyTorch 模型。
        target: 相对于模型根节点的模块路径。
        spec: 与张量网络表示对应的分解配置。
        decomposition_backend: 将稠密权重分解为 artifact 的后端。
        execution_backend: 根据 artifact 构造压缩模型层的后端。
        trainable: 压缩模型层的参数是否参与训练。
        decomposition_dtype: 可选分解浮点类型；省略时使用原权重类型，结果恢复为原层类型。

    返回：
        包含通用 artifact 和模型替换记录的单层压缩结果。
    """

    if decomposition_dtype is not None and not decomposition_dtype.is_floating_point:
        raise ValueError("decomposition_dtype must be floating-point")
    linear = find_linear(model, target)
    weight = linear.weight.detach()
    decomposition_weight = (
        weight.to(dtype=decomposition_dtype)
        if decomposition_dtype is not None
        else weight
    )
    tn_artifact = decomposition_backend.decompose(decomposition_weight, spec)
    if decomposition_dtype is not None:
        tn_artifact = tn_artifact.to(device=weight.device, dtype=weight.dtype)
    compressed = execution_backend.build_linear(
        tn_artifact,
        trainable=trainable,
    )
    replacement = replace_linear(model, target, compressed)
    return LinearCompressionResult(tn_artifact, replacement)


def compress_model(
    model: nn.Module,
    plan: CompressionPlan,
    *,
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    trainable: bool,
    decomposition_dtype: torch.dtype | None = None,
) -> ModelCompressionResult:
    """按照压缩计划分解并替换多个 Linear。

    参数：
        model: 包含全部目标 Linear 的 PyTorch 模型。
        plan: 记录目标路径、表示名称和各层结构配置的压缩计划。
        decomposition_backends: 按表示名称提供分解后端的映射。
        execution_backends: 按表示名称提供执行后端的映射。
        trainable: 所有压缩模型层的参数是否参与训练。
        decomposition_dtype: 各层分解使用的可选浮点类型；结果恢复为各层原始类型。

    返回：
        与计划目标顺序一致的逐层压缩结果。

    异常：
        ValueError: 缺少所需 backend 或 backend 表示不一致时抛出。
        TypeError: 任一目标不是 ``torch.nn.Linear`` 时抛出。
        Exception: 压缩过程中发生错误时，恢复已替换层后继续抛出原始异常。
    """

    representations = {target.representation for target in plan.targets}
    for representation in representations:
        if representation not in decomposition_backends:
            raise ValueError(
                f"missing decomposition backend for representation {representation!r}"
            )
        if representation not in execution_backends:
            raise ValueError(
                f"missing execution backend for representation {representation!r}"
            )
        for role, backend in (
            ("decomposition", decomposition_backends[representation]),
            ("execution", execution_backends[representation]),
        ):
            if backend.representation.strip().lower() != representation:
                raise ValueError(
                    f"{role} backend for {representation!r} declares representation "
                    f"{backend.representation!r}"
                )
    for target in plan.targets:
        find_linear(model, target.module_path)

    layer_results: list[LinearCompressionResult] = []
    try:
        for target in plan.targets:
            layer_results.append(
                compress_linear(
                    model,
                    target.module_path,
                    target.spec,
                    decomposition_backend=decomposition_backends[
                        target.representation
                    ],
                    execution_backend=execution_backends[target.representation],
                    trainable=trainable,
                    decomposition_dtype=decomposition_dtype,
                )
            )
    except Exception:
        for result in reversed(layer_results):
            restore_linear(model, result.replacement)
        raise
    return ModelCompressionResult(tuple(layer_results))


def restore_compressed_model(
    model: nn.Module,
    result: ModelCompressionResult,
) -> None:
    """释放模型级压缩层并恢复原始 Linear。

    参数：
        model: 当前安装了批量压缩层的 PyTorch 模型。
        result: ``compress_model`` 返回的逐层压缩结果。
    """

    for layer_result in reversed(result.layer_results):
        restore_linear(model, layer_result.replacement)
