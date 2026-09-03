"""定义规范 MPO 格式及与后端无关的数学操作。

本模块集中规定 MPO 的结构配置、core 布局、稠密权重形状和 artifact 格式，并实现
矩阵张量化与重建。所有计算后端复用这些定义，不各自维护另一套 MPO 规则。

主要内容：
- ``MPOSpec``：校验维度与 ranks，并计算稠密和 MPO 参数量。
- ``validate_mpo_cores``、``validate_mpo_weight``：校验 cores 和稠密权重。
- ``make_mpo_artifact``、``parse_mpo_artifact``：构造和解析规范 MPO artifact。
- ``tensorize_matrix``、``detensorize_matrix``：转换稠密矩阵与 MPO 张量布局。
- ``reconstruct_mpo``：收缩规范 cores 并恢复稠密权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .artifact import TensorNetworkArtifact

MPO_REPRESENTATION = "mpo"


def _positive_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    """将 mode 或 rank 值转换为非空正整数元组。

    参数：
        values: 待规范化的整数值。
        name: 校验错误信息中使用的字段名。

    返回：
        规范化后的整数元组。
    """

    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


@dataclass(frozen=True)
class MPOSpec:
    """定义规范四维 MPO cores 的 modes 和 ranks。"""

    out_modes: tuple[int, ...]
    in_modes: tuple[int, ...]
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        """规范化 modes，并验证每个请求的 rank 均可实现。"""

        object.__setattr__(
            self, "out_modes", _positive_tuple(self.out_modes, "out_modes")
        )
        object.__setattr__(
            self, "in_modes", _positive_tuple(self.in_modes, "in_modes")
        )
        object.__setattr__(self, "ranks", _positive_tuple(self.ranks, "ranks"))
        if len(self.out_modes) != len(self.in_modes):
            raise ValueError("out_modes and in_modes must have the same length")
        if len(self.ranks) != self.order + 1:
            raise ValueError("ranks must contain order + 1 entries")
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("MPO boundary ranks must both be 1")
        for index, (rank, maximum) in enumerate(
            zip(self.ranks[1:-1], self.maximum_internal_ranks, strict=True),
            start=1,
        ):
            if rank > maximum:
                raise ValueError(f"rank r{index}={rank} exceeds maximum {maximum}")

    @property
    def order(self) -> int:
        """返回 MPO cores 的数量。"""

        return len(self.out_modes)

    @property
    def out_features(self) -> int:
        """返回稠密矩阵的输出维度。"""

        return prod(self.out_modes)

    @property
    def in_features(self) -> int:
        """返回稠密矩阵的输入维度。"""

        return prod(self.in_modes)

    @property
    def physical_modes(self) -> tuple[int, ...]:
        """返回成对的输出与输入 mode 大小。"""

        return tuple(
            out_mode * in_mode
            for out_mode, in_mode in zip(
                self.out_modes, self.in_modes, strict=True
            )
        )

    @property
    def maximum_internal_ranks(self) -> tuple[int, ...]:
        """返回每条内部 bond 可采用的最大矩阵 rank。"""

        return tuple(
            min(prod(self.physical_modes[:index]), prod(self.physical_modes[index:]))
            for index in range(1, self.order)
        )

    @property
    def core_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        """返回规范 [left, out, in, right] core 形状。"""

        return tuple(
            (
                self.ranks[index],
                self.out_modes[index],
                self.in_modes[index],
                self.ranks[index + 1],
            )
            for index in range(self.order)
        )

    @property
    def num_parameters(self) -> int:
        """返回 MPO 标量参数数量。"""

        return sum(prod(shape) for shape in self.core_shapes)

    @property
    def dense_num_parameters(self) -> int:
        """返回稠密矩阵的标量参数数量。"""

        return self.out_features * self.in_features

    @property
    def compression_ratio(self) -> float:
        """返回稠密参数量与 MPO 参数量的比值。"""

        return self.dense_num_parameters / self.num_parameters

    @classmethod
    def full_rank(
        cls,
        out_modes: Sequence[int],
        in_modes: Sequence[int],
    ) -> "MPOSpec":
        """创建内部 ranks 足以完整保留任意矩阵的结构配置。

        参数：
            out_modes: 稠密输出维度的因子分解。
            in_modes: 稠密输入维度的因子分解。

        返回：
            满 rank 的 MPO 结构配置。
        """

        normalized_out = _positive_tuple(out_modes, "out_modes")
        normalized_in = _positive_tuple(in_modes, "in_modes")
        if len(normalized_out) != len(normalized_in):
            raise ValueError("out_modes and in_modes must have the same length")
        physical_modes = tuple(
            out_mode * in_mode
            for out_mode, in_mode in zip(
                normalized_out, normalized_in, strict=True
            )
        )
        internal_ranks = tuple(
            min(prod(physical_modes[:index]), prod(physical_modes[index:]))
            for index in range(1, len(physical_modes))
        )
        return cls(normalized_out, normalized_in, (1, *internal_ranks, 1))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "MPOSpec":
        """从 artifact 元数据创建 MPO 结构配置。

        参数：
            metadata: 包含输出 modes、输入 modes 和 ranks 的映射。

        返回：
            解析得到的 MPO 结构配置。
        """

        return cls(
            out_modes=tuple(metadata["out_modes"]),
            in_modes=tuple(metadata["in_modes"]),
            ranks=tuple(metadata["ranks"]),
        )

    def to_metadata(self) -> dict[str, list[int]]:
        """序列化 MPO artifact 使用的结构字段。"""

        return {
            "out_modes": list(self.out_modes),
            "in_modes": list(self.in_modes),
            "ranks": list(self.ranks),
        }


def validate_mpo_cores(cores: Sequence[Tensor], spec: MPOSpec) -> None:
    """校验规范 MPO cores 的形状、数据类型和设备。

    参数：
        cores: 按 [left_rank, out_mode, in_mode, right_rank] 排列的 cores。
        spec: cores 应满足的 MPO 结构配置。
    """

    if len(cores) != spec.order:
        raise ValueError(f"expected {spec.order} cores, got {len(cores)}")
    for index, (core, shape) in enumerate(
        zip(cores, spec.core_shapes, strict=True)
    ):
        if tuple(core.shape) != shape:
            raise ValueError(
                f"core {index} expected shape {shape}, got {tuple(core.shape)}"
            )
        if not core.is_floating_point():
            raise TypeError(f"core {index} must be floating point")
    if len({core.dtype for core in cores}) != 1:
        raise ValueError("all MPO cores must have the same dtype")
    if len({core.device for core in cores}) != 1:
        raise ValueError("all MPO cores must be on the same device")


def validate_mpo_weight(weight: Tensor, spec: MPOSpec) -> None:
    """校验待分解稠密权重的形状和数据类型。

    参数：
        weight: 待分解的稠密矩阵。
        spec: 请求使用的 MPO 结构。
    """

    expected_shape = (spec.out_features, spec.in_features)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"expected weight shape {expected_shape}, got {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")


def make_mpo_artifact(
    spec: MPOSpec,
    cores: Sequence[Tensor],
) -> TensorNetworkArtifact:
    """将规范 MPO cores 封装为与具体表示类型无关的 artifact。

    参数：
        spec: MPO 的 modes 和 ranks。
        cores: 规范 MPO cores。

    返回：
        标记为 MPO 的通用 artifact。
    """

    validate_mpo_cores(cores, spec)
    return TensorNetworkArtifact(
        representation=MPO_REPRESENTATION,
        metadata=spec.to_metadata(),
        tensors={f"core_{index}": core for index, core in enumerate(cores)},
    )


def parse_mpo_artifact(
    artifact: TensorNetworkArtifact,
) -> tuple[MPOSpec, tuple[Tensor, ...]]:
    """解析并校验通用 artifact 中的 MPO 内容。

    参数：
        artifact: 预期包含 MPO 的 artifact。

    返回：
        MPO 结构配置和按顺序排列的规范 cores。
    """

    if artifact.representation != MPO_REPRESENTATION:
        raise ValueError(
            f"expected {MPO_REPRESENTATION!r}, got {artifact.representation!r}"
        )
    spec = MPOSpec.from_metadata(artifact.metadata)
    expected_names = tuple(f"core_{index}" for index in range(spec.order))
    if set(artifact.tensors) != set(expected_names):
        raise ValueError(f"MPO tensors must be named {list(expected_names)!r}")
    cores = tuple(artifact.tensors[name] for name in expected_names)
    validate_mpo_cores(cores, spec)
    return spec, cores


def tensorize_matrix(weight: Tensor, spec: MPOSpec) -> Tensor:
    """为 MPO 分解交错排列输出和输入 modes。

    参数：
        weight: 形状为 [out_features, in_features] 的稠密矩阵。
        spec: 目标 MPO 结构。

    返回：
        各坐标轴为成对物理 modes 的张量。
    """

    expected_shape = (spec.out_features, spec.in_features)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"expected weight shape {expected_shape}, got {tuple(weight.shape)}"
        )
    shaped = weight.reshape(*spec.out_modes, *spec.in_modes)
    permutation = tuple(
        axis
        for pair in zip(
            range(spec.order),
            range(spec.order, 2 * spec.order),
            strict=True,
        )
        for axis in pair
    )
    return shaped.permute(permutation).contiguous().reshape(*spec.physical_modes)


def detensorize_matrix(tensor: Tensor, spec: MPOSpec) -> Tensor:
    """将成对物理 modes 转换回稠密矩阵。

    参数：
        tensor: 每个 MPO core 对应一个成对物理轴的张量。
        spec: 描述稠密维度的 MPO 结构。

    返回：
        形状为 [out_features, in_features] 的稠密矩阵。
    """

    expected_shape = spec.physical_modes
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"expected tensor shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    interleaved_shape = tuple(
        value
        for pair in zip(spec.out_modes, spec.in_modes, strict=True)
        for value in pair
    )
    permutation = tuple(range(0, 2 * spec.order, 2)) + tuple(
        range(1, 2 * spec.order, 2)
    )
    return (
        tensor.reshape(*interleaved_shape)
        .permute(permutation)
        .contiguous()
        .reshape(spec.out_features, spec.in_features)
    )


def reconstruct_mpo(artifact: TensorNetworkArtifact) -> Tensor:
    """从规范 MPO artifact 重建稠密矩阵。

    参数：
        artifact: 包含规范 MPO cores 的通用 artifact。

    返回：
        重建后的稠密矩阵。
    """

    spec, cores = parse_mpo_artifact(artifact)
    state = cores[0]
    for core in cores[1:]:
        state = torch.tensordot(state, core, dims=([-1], [0]))
    paired = state.squeeze(0).squeeze(-1).reshape(*spec.physical_modes)
    return detensorize_matrix(paired, spec)
