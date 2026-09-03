"""定义与张量网络表示无关的后端公共接口。

本模块规定计算库适配器如何报告能力与环境状态，以及如何把稠密权重分解为 artifact、
把 artifact 构造成可执行 PyTorch 模块。结构配置和张量布局属于 representations 层，
有状态模型层的公共行为属于 ``nn`` 层。

主要内容：
- ``BackendCapabilities``：描述分解、训练和推理三类能力。
- ``BackendProbe``：描述依赖是否可用、版本和不可用原因。
- ``TensorNetworkBackend``：无状态适配器基类，负责分解和模型层构造，不保存层参数。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import Tensor

from ..representations import TensorNetworkArtifact
from ..nn.base import TensorNetworkLinear

SpecT = TypeVar("SpecT")


@dataclass(frozen=True)
class BackendCapabilities:
    """描述后端支持的操作和硬件要求。"""

    decomposition: bool
    inference: bool
    training: bool
    requires_cuda: bool = False


@dataclass(frozen=True)
class BackendProbe:
    """描述后端在当前环境中是否可用。"""

    available: bool
    reason: str | None = None


class TensorNetworkBackend(ABC, Generic[SpecT]):
    """定义单个计算库的分解和线性层构造接口。"""

    provider: str
    representation: str

    @property
    def version(self) -> str:
        """返回已安装后端的版本。"""

        return "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """返回当前后端支持的操作。"""

    def probe(self) -> BackendProbe:
        """检查当前后端能否在现有环境中运行。"""

        return BackendProbe(available=True)

    def decompose(
        self,
        weight: Tensor,
        spec: SpecT,
    ) -> TensorNetworkArtifact:
        """将稠密权重分解为规范 artifact。

        参数：
            weight: 由具体后端解释的稠密权重。
            spec: 与张量网络表示对应的分解配置。

        返回：
            规范的张量网络 artifact。

        异常：
            NotImplementedError: 当前后端不支持分解时抛出。
        """

        raise NotImplementedError(
            f"{self.provider}/{self.representation} does not support decomposition"
        )

    @abstractmethod
    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> TensorNetworkLinear:
        """构造无 bias 的张量网络线性层运行时。

        参数：
            artifact: 规范的张量网络表示 artifact。
            trainable: 运行时参数是否需要参与训练。

        返回：
            后端特有的张量网络线性层模块。
        """
