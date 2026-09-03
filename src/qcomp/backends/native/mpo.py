"""使用原生 PyTorch 实现 MPO 分解和执行。

本模块对稠密 Linear 权重执行 TT-SVD，将结果转换为规范 MPO cores，并直接使用
PyTorch 张量操作完成收缩。该实现支持自动求导训练和推理，不依赖可选第三方库。

主要内容：
- ``_tt_svd``：把张量化矩阵顺序分解为 TT cores。
- ``native_mpo_forward``：使用原生 PyTorch 收缩规范 MPO cores。
- ``NativeMPOLinear``：有状态模型层，持有 cores 并执行 forward。
- ``NativeMPOBackend``：无状态适配器，报告能力并负责分解和模型层构造。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...nn.mpo import CanonicalMPOLinearBase, MPOLinearBase
from ...representations import MPOSpec, TensorNetworkArtifact, make_mpo_artifact
from ...representations.mpo import tensorize_matrix, validate_mpo_weight
from ..base import BackendCapabilities, TensorNetworkBackend


def _tt_svd(weight: Tensor, spec: MPOSpec) -> tuple[Tensor, ...]:
    """使用顺序截断 SVD 分解稠密矩阵。

    参数：
        weight: 形状为 [out_features, in_features] 的稠密矩阵。
        spec: 请求使用的 MPO modes 和 ranks。

    返回：
        规范 MPO cores。
    """

    remainder = tensorize_matrix(weight, spec)
    cores: list[Tensor] = []
    left_rank = 1
    for index in range(spec.order - 1):
        matrix = remainder.reshape(left_rank * spec.physical_modes[index], -1)
        target_rank = spec.ranks[index + 1]
        left, singular_values, right = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        cores.append(
            left[:, :target_rank]
            .reshape(
                left_rank,
                spec.out_modes[index],
                spec.in_modes[index],
                target_rank,
            )
            .contiguous()
        )
        remainder = (
            singular_values[:target_rank].unsqueeze(1) * right[:target_rank]
        ).reshape(target_rank, *spec.physical_modes[index + 1 :])
        left_rank = target_rank
    cores.append(
        remainder.reshape(
            left_rank,
            spec.out_modes[-1],
            spec.in_modes[-1],
            1,
        ).contiguous()
    )
    return tuple(cores)


def native_mpo_forward(
    flat_input: Tensor,
    cores: tuple[Tensor, ...],
    spec: MPOSpec,
) -> Tensor:
    """使用规范 MPO cores 收缩二维输入矩阵。

    参数：
        flat_input: 形状为 [tokens, in_features] 的输入。
        cores: 规范 MPO cores。
        spec: MPO 的 modes 和 ranks。

    返回：
        形状为 [tokens, out_features] 的输出。
    """

    tokens = flat_input.shape[0]
    state = flat_input.reshape(tokens, *spec.in_modes)
    state = torch.tensordot(state, cores[-1].squeeze(-1), dims=([-1], [2]))
    for index in range(spec.order - 2, -1, -1):
        remaining_modes = index + 1
        state = torch.tensordot(
            state,
            cores[index],
            dims=([remaining_modes, remaining_modes + 1], [2, 3]),
        )
        accumulated_outputs = spec.order - index - 1
        raw_output_start = remaining_modes
        rank_axis = raw_output_start + accumulated_outputs
        current_output_axis = rank_axis + 1
        permutation = (
            [0]
            + list(range(1, remaining_modes))
            + [rank_axis, current_output_axis]
            + list(
                range(
                    raw_output_start,
                    raw_output_start + accumulated_outputs,
                )
            )
        )
        state = state.permute(permutation).contiguous()
    return state.squeeze(1).reshape(tokens, spec.out_features)


class NativeMPOLinear(CanonicalMPOLinearBase):
    """使用原生 PyTorch 收缩执行规范 MPO cores。"""

    @property
    def backend_name(self) -> str:
        """返回稳定的 native 后端名称。"""

        return "native"

    def _contract(self, flat_input: Tensor) -> Tensor:
        """使用原生 PyTorch 张量操作收缩输入。

        参数：
            flat_input: 形状为 [tokens, in_features] 的输入。

        返回：
            形状为 [tokens, out_features] 的输出。
        """

        return native_mpo_forward(flat_input, self.core_tensors(), self.spec)


class NativeMPOBackend(TensorNetworkBackend[MPOSpec]):
    """提供原生 PyTorch 分解、推理和训练能力。"""

    provider = "native"
    representation = "mpo"

    @property
    def version(self) -> str:
        """返回当前后端使用的 PyTorch 版本。"""

        return torch.__version__

    @property
    def capabilities(self) -> BackendCapabilities:
        """返回 native 后端能力。"""

        return BackendCapabilities(
            decomposition=True,
            inference=True,
            training=True,
        )

    def decompose(
        self,
        weight: Tensor,
        spec: MPOSpec,
    ) -> TensorNetworkArtifact:
        """使用原生 TT-SVD 分解稠密权重。

        参数：
            weight: 形状为 [out_features, in_features] 的稠密矩阵。
            spec: 请求使用的 MPO modes 和 ranks。

        返回：
            规范 MPO artifact。
        """

        validate_mpo_weight(weight, spec)
        return make_mpo_artifact(spec, _tt_svd(weight, spec))

    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> MPOLinearBase:
        """构造原生的无 bias MPO 线性层模块。

        参数：
            artifact: 规范 MPO artifact。
            trainable: cores 是否需要梯度。

        返回：
            原生 MPO 运行时。
        """

        return NativeMPOLinear(artifact, trainable=trainable)
