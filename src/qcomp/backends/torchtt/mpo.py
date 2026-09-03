"""为 MPO 适配 torchTT 分解和 LinearLayerTT 执行。

本模块使用 torchTT 分解稠密权重，在 torchTT cores 与 qcomp artifact 之间转换，并
构造可训练、可推理的 LinearLayerTT。内部零 bias 仅用于适配第三方容器，对外保持
无 bias Linear 语义。

主要内容：
- ``_library``：延迟导入 torchTT。
- ``TorchTTMPOLinear``：有状态模型层，持有 torchTT cores 并执行 forward。
- ``TorchTTMPOBackend``：无状态适配器，报告能力并负责分解和模型层构造。
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from typing import Any

import torch
from torch import Tensor, nn

from ...nn.mpo import MPOLinearBase
from ...representations import (
    MPOSpec,
    TensorNetworkArtifact,
    make_mpo_artifact,
    parse_mpo_artifact,
)
from ...representations.mpo import validate_mpo_weight
from ..base import (
    BackendCapabilities,
    BackendProbe,
    TensorNetworkBackend,
)


def _library() -> Any:
    """仅在选择当前后端时导入 torchTT。"""

    import torchtt

    return torchtt


class TorchTTMPOLinear(MPOLinearBase):
    """使用 torchTT LinearLayerTT 执行规范 MPO cores。"""

    def __init__(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> None:
        """根据规范 artifact 构造 torchTT 层。

        参数：
            artifact: 规范 MPO artifact。
            trainable: MPO cores 是否需要梯度。
        """

        spec, cores = parse_mpo_artifact(artifact)
        super().__init__(spec)
        torchtt = _library()
        linear = torchtt.nn.LinearLayerTT(
            list(spec.in_modes),
            list(spec.out_modes),
            list(spec.ranks),
            dtype=cores[0].dtype,
        )
        linear.cores = nn.ParameterList(
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        )
        linear.bias = nn.Parameter(
            torch.zeros(
                spec.out_modes,
                device=cores[0].device,
                dtype=cores[0].dtype,
            ),
            requires_grad=False,
        )
        self.linear = linear

    @property
    def backend_name(self) -> str:
        """返回稳定的 torchTT 后端名称。"""

        return "torchtt"

    def core_tensors(self) -> tuple[Tensor, ...]:
        """按规范顺序返回 torchTT cores。"""

        return tuple(self.linear.cores)

    def _contract(self, flat_input: Tensor) -> Tensor:
        """对张量化输入执行 torchTT LinearLayerTT。

        参数：
            flat_input: 形状为 [tokens, in_features] 的输入。

        返回：
            形状为 [tokens, out_features] 的输出。
        """

        tensorized = flat_input.reshape(flat_input.shape[0], *self.spec.in_modes)
        return self.linear(tensorized).reshape(
            flat_input.shape[0],
            self.spec.out_features,
        )


class TorchTTMPOBackend(TensorNetworkBackend[MPOSpec]):
    """提供 torchTT 分解、推理和训练能力。"""

    provider = "torchtt"
    representation = "mpo"

    @property
    def version(self) -> str:
        """返回已安装的 torchTT 版本。"""

        return version("torchTT")

    @property
    def capabilities(self) -> BackendCapabilities:
        """返回 torchTT 后端能力。"""

        return BackendCapabilities(
            decomposition=True,
            inference=True,
            training=True,
        )

    def probe(self) -> BackendProbe:
        """检查 torchTT 是否可导入。"""

        available = importlib.util.find_spec("torchtt") is not None
        return BackendProbe(
            available=available,
            reason=None if available else "missing package: torchTT",
        )

    def decompose(
        self,
        weight: Tensor,
        spec: MPOSpec,
    ) -> TensorNetworkArtifact:
        """使用 torchTT TT-SVD 分解稠密矩阵。

        参数：
            weight: 形状为 [out_features, in_features] 的稠密矩阵。
            spec: 请求使用的 MPO modes 和最大 ranks。

        返回：
            规范 MPO artifact。
        """

        validate_mpo_weight(weight, spec)
        torchtt = _library()
        shape = list(zip(spec.out_modes, spec.in_modes, strict=True))
        result = torchtt.TT(
            weight,
            shape,
            eps=0.0,
            rmax=list(spec.ranks),
        )
        cores = tuple(core.contiguous() for core in result.cores)
        return make_mpo_artifact(spec, cores)

    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> MPOLinearBase:
        """构造无 bias 的 torchTT MPO 线性层模块。

        参数：
            artifact: 规范 MPO artifact。
            trainable: cores 是否需要梯度。

        返回：
            torchTT MPO 运行时。
        """

        return TorchTTMPOLinear(artifact, trainable=trainable)
