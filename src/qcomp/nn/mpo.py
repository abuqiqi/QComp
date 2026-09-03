"""提供 MPO PyTorch 模块共享的运行行为。

本模块解析规范 MPO artifact，统一处理无 bias Linear 的输入输出形状，并为直接保存
规范 cores 的实现提供参数注册和 artifact 导出能力。具体 core 收缩由 backends 层
中的计算库适配器实现。

主要内容：
- ``MPOLinearBase``：解析 MPO 元数据、检查输入形状并恢复输出形状。
- ``CanonicalMPOLinearBase``：把规范 cores 注册为参数并导出最新 artifact。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..representations import (
    MPOSpec,
    TensorNetworkArtifact,
    make_mpo_artifact,
    parse_mpo_artifact,
)
from .base import TensorNetworkLinear


class MPOLinearBase(TensorNetworkLinear, ABC):
    """为 MPO 线性层提供统一的形状处理和 artifact 导出。"""

    def __init__(self, spec: MPOSpec) -> None:
        """初始化 MPO 运行时的公共维度。

        参数：
            spec: 运行时采用的 MPO 结构配置。
        """

        super().__init__()
        self.spec = spec

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """返回稳定的后端名称。"""

    @abstractmethod
    def core_tensors(self) -> tuple[Tensor, ...]:
        """返回规范顺序的 MPO core 张量。"""

    @abstractmethod
    def _contract(self, flat_input: Tensor) -> Tensor:
        """使用 MPO cores 收缩二维输入矩阵。

        参数：
            flat_input: 形状为 [tokens, in_features] 的输入。

        返回：
            形状为 [tokens, out_features] 的输出。
        """

    def export_artifact(self) -> TensorNetworkArtifact:
        """将当前 MPO 参数导出为规范 artifact。"""

        cores = tuple(core.detach() for core in self.core_tensors())
        return make_mpo_artifact(self.spec, cores)

    def forward(self, inputs: Tensor) -> Tensor:
        """执行无 bias 的 MPO 线性变换。

        参数：
            inputs: 最后一维等于 in_features 的输入张量。

        返回：
            保留输入前导维度且最后一维为 out_features 的张量。
        """

        if inputs.shape[-1] != self.spec.in_features:
            raise ValueError(
                f"expected last dimension {self.spec.in_features}, "
                f"got {inputs.shape[-1]}"
            )
        leading_shape = inputs.shape[:-1]
        flat_input = inputs.reshape(-1, self.spec.in_features)
        output = self._contract(flat_input)
        return output.reshape(*leading_shape, self.spec.out_features)


class CanonicalMPOLinearBase(MPOLinearBase):
    """为直接保存规范 MPO cores 的 runtime 提供参数管理。"""

    def __init__(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> None:
        """使用规范 artifact 中的张量构造运行时。

        参数：
            artifact: 规范 MPO artifact。
            trainable: MPO cores 是否需要梯度。
        """

        spec, cores = parse_mpo_artifact(artifact)
        super().__init__(spec)
        self.cores = nn.ParameterList(
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        )

    def core_tensors(self) -> tuple[Tensor, ...]:
        """返回直接保存的规范 core 参数。"""

        return tuple(self.cores)
