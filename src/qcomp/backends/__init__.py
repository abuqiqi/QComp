"""公开计算后端接口和 Provider 懒加载查询。

本包汇总与具体表示无关的 backend 类型，并通过 registry 根据 Provider 与张量网络
表示定位具体适配器。可选计算库只在请求对应组合时加载。

主要内容：
- ``BackendCapabilities``：记录后端是否支持分解、训练和推理。
- ``BackendProbe``：记录后端可用性、版本和不可用原因。
- ``TensorNetworkBackend``：定义无状态后端适配器的通用接口。
- ``get_backend``、``list_backends``：查询已注册的 Provider 与表示组合。
"""

from .base import (
    BackendCapabilities,
    BackendProbe,
    TensorNetworkBackend,
)
from .registry import get_backend, list_backends

__all__ = [
    "BackendCapabilities",
    "BackendProbe",
    "TensorNetworkBackend",
    "get_backend",
    "list_backends",
]
