"""按张量网络表示名称查找并执行稠密重建。

本模块维护项目内置表示与重建函数之间的映射，并根据通用 artifact 的
``representation`` 字段延迟加载对应实现。它只负责表示级数学操作的分派，不依赖
计算 backend，也不提供外部插件注册。

主要内容：
- ``reconstruct_tensor``：把已注册表示的通用 artifact 重建为稠密张量。
"""

from __future__ import annotations

import importlib

from torch import Tensor

from .artifact import TensorNetworkArtifact

_RECONSTRUCTORS = {
    "mpo": (".mpo", "reconstruct_mpo"),
}


def reconstruct_tensor(tn_artifact: TensorNetworkArtifact) -> Tensor:
    """根据 artifact 的表示类型重建稠密张量。

    参数：
        tn_artifact: 包含已注册张量网络表示的通用 artifact。

    返回：
        对应张量网络表示重建得到的稠密张量。

    异常：
        ValueError: artifact 的表示类型没有注册重建函数时抛出。
    """

    try:
        module_name, function_name = _RECONSTRUCTORS[tn_artifact.representation]
    except KeyError as error:
        raise ValueError(
            "no reconstructor registered for representation: "
            f"{tn_artifact.representation!r}"
        ) from error
    module = importlib.import_module(module_name, package=__package__)
    reconstruct = getattr(module, function_name)
    return reconstruct(tn_artifact)
