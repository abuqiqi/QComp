"""按 Provider 与张量网络表示注册和延迟构造计算后端。

本模块将 ``(provider, representation)`` 组合映射到具体适配器，不在 qcomp 启动时
导入可选计算库。调用者可以列出某种表示支持的 Provider，或取得一个已注册组合。

主要内容：
- ``list_backends``：列出指定张量网络表示已注册的 Provider。
- ``get_backend``：延迟导入并实例化指定 Provider 与表示组合。
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import TensorNetworkBackend

_BACKENDS = {
    ("native", "mpo"): (".native.mpo", "NativeMPOBackend"),
    ("tensorly", "mpo"): (".tensorly.mpo", "TensorLyMPOBackend"),
    ("torchtt", "mpo"): (".torchtt.mpo", "TorchTTMPOBackend"),
    ("cutensornet", "mpo"): (".cutensornet.mpo", "CuTensorNetMPOBackend"),
}


def list_backends(representation: str) -> tuple[str, ...]:
    """返回支持指定张量网络表示的 Provider 名称。

    参数：
        representation: 待查询的张量网络表示名称。

    返回：
        按注册顺序排列的 Provider 名称。
    """

    normalized = representation.strip().lower()
    return tuple(
        provider
        for provider, registered_representation in _BACKENDS
        if registered_representation == normalized
    )


def get_backend(
    provider: str,
    representation: str,
) -> TensorNetworkBackend[Any]:
    """根据 Provider 与表示名称创建后端适配器。

    参数：
        provider: 计算库 Provider 名称。
        representation: 张量网络表示名称。

    返回：
        新建的无状态后端适配器。

    异常：
        ValueError: 请求的 Provider 与表示组合未注册时抛出。
    """

    key = (provider.strip().lower(), representation.strip().lower())
    try:
        module_name, class_name = _BACKENDS[key]
    except KeyError as error:
        raise ValueError(f"unregistered backend combination: {key!r}") from error
    module = importlib.import_module(module_name, package=__package__)
    backend_type = getattr(module, class_name)
    return backend_type()
