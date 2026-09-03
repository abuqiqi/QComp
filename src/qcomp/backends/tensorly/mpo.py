"""为 MPO 适配 TensorLy 分解和 TensorLy-Torch 执行。

本模块使用 TensorLy 分解稠密权重，将结果转换为 qcomp 的规范 artifact，并使用
TensorLy-Torch 构造可训练、可推理的模型层。模块只负责 TensorLy 与 MPO 这一组合。

主要内容：
- ``_libraries``：延迟导入 TensorLy 和 TensorLy-Torch。
- ``TensorLyMPOLinear``：有状态模型层，持有 factors 并执行 forward。
- ``TensorLyMPOBackend``：无状态适配器，报告能力并负责分解和模型层构造。
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from typing import Any

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


def _libraries() -> tuple[Any, Any, Any, Any]:
    """仅在选择当前后端时导入 TensorLy 和 TensorLy-Torch。"""

    import tensorly
    from tensorly.decomposition import tensor_train_matrix
    from tltorch import BlockTT, FactorizedLinear

    return tensorly, tensor_train_matrix, BlockTT, FactorizedLinear


class TensorLyMPOLinear(MPOLinearBase):
    """使用 TensorLy-Torch FactorizedLinear 执行规范 MPO cores。"""

    def __init__(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> None:
        """根据规范 artifact 构造 TensorLy-Torch 对象。

        参数：
            artifact: 规范 MPO artifact。
            trainable: MPO factors 是否需要梯度。
        """

        spec, cores = parse_mpo_artifact(artifact)
        super().__init__(spec)
        _, _, BlockTT, FactorizedLinear = _libraries()
        parameters = [
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        ]
        factorization = BlockTT(
            parameters,
            tensorized_shape=(spec.out_modes, spec.in_modes),
            rank=spec.ranks,
        )
        self.linear = FactorizedLinear(
            in_tensorized_features=spec.in_modes,
            out_tensorized_features=spec.out_modes,
            bias=False,
            factorization=factorization,
            implementation="factorized",
            checkpointing=False,
            device=cores[0].device,
            dtype=cores[0].dtype,
        )

    @property
    def backend_name(self) -> str:
        """返回稳定的 TensorLy 后端名称。"""

        return "tensorly"

    def core_tensors(self) -> tuple[Tensor, ...]:
        """按规范 core 顺序返回 TensorLy-Torch factors。"""

        return tuple(self.linear.weight.factors)

    def _contract(self, flat_input: Tensor) -> Tensor:
        """执行 TensorLy-Torch FactorizedLinear。

        参数：
            flat_input: 形状为 [tokens, in_features] 的输入。

        返回：
            形状为 [tokens, out_features] 的输出。
        """

        return self.linear(flat_input)


class TensorLyMPOBackend(TensorNetworkBackend[MPOSpec]):
    """提供 TensorLy 分解和 TensorLy-Torch 运行时操作。"""

    provider = "tensorly"
    representation = "mpo"

    @property
    def version(self) -> str:
        """返回 TensorLy 和 TensorLy-Torch 版本。"""

        return (
            f"tensorly={version('tensorly')},"
            f"tensorly-torch={version('tensorly-torch')}"
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """返回 TensorLy 后端能力。"""

        return BackendCapabilities(
            decomposition=True,
            inference=True,
            training=True,
        )

    def probe(self) -> BackendProbe:
        """检查 TensorLy 和 TensorLy-Torch 是否可导入。"""

        missing = [
            name
            for name in ("tensorly", "tltorch")
            if importlib.util.find_spec(name) is None
        ]
        reason = f"missing packages: {', '.join(missing)}" if missing else None
        return BackendProbe(available=not missing, reason=reason)

    def decompose(
        self,
        weight: Tensor,
        spec: MPOSpec,
    ) -> TensorNetworkArtifact:
        """使用 TensorLy tensor_train_matrix 分解稠密矩阵。

        参数：
            weight: 形状为 [out_features, in_features] 的稠密矩阵。
            spec: 请求使用的 MPO modes 和 ranks。

        返回：
            规范 MPO artifact。
        """

        validate_mpo_weight(weight, spec)
        tensorly, tensor_train_matrix, _, _ = _libraries()
        tensor = weight.reshape(*spec.out_modes, *spec.in_modes)
        with tensorly.backend_context("pytorch"):
            result = tensor_train_matrix(tensor, rank=spec.ranks)
        cores = tuple(core.contiguous() for core in result.factors)
        return make_mpo_artifact(spec, cores)

    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> MPOLinearBase:
        """构造无 bias 的 TensorLy-Torch MPO 线性层模块。

        参数：
            artifact: 规范 MPO artifact。
            trainable: factors 是否需要梯度。

        返回：
            TensorLy-Torch MPO 运行时。
        """

        return TensorLyMPOLinear(artifact, trainable=trainable)
