"""定义张量网络 PyTorch 线性模块的通用接口。

本模块规定可执行压缩线性层如何导出当前参数 artifact 和释放运行资源。每个实例对应
模型中的一个压缩层并独立持有状态，而无状态 backend 只负责创建这些实例。

主要内容：
- ``TensorNetworkLinear``：要求模型层实现 artifact 导出，并提供统一资源释放入口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..representations import TensorNetworkArtifact


class TensorNetworkLinear(nn.Module, ABC):
    """定义张量网络线性模块共有的生命周期操作。"""

    @abstractmethod
    def export_artifact(self) -> TensorNetworkArtifact:
        """将当前模块参数导出为规范 artifact。"""

    def close(self) -> None:
        """释放计算库特有的运行资源。"""
