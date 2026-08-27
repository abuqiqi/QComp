"""Backend-independent TT-matrix operations."""

from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor
from .spec import TTMatrixSpec


def validate_cores(cores: Sequence[Tensor], spec: TTMatrixSpec) -> None:
    if len(cores) != spec.order:
        raise ValueError(f"expected {spec.order} cores, got {len(cores)}")
    for index, (core, shape) in enumerate(zip(cores, spec.core_shapes, strict=True)):
        if not isinstance(core, Tensor):
            raise TypeError(f"core {index} must be a torch.Tensor")
        if tuple(core.shape) != shape:
            raise ValueError(
                f"core {index} expected shape {shape}, got {tuple(core.shape)}"
            )
        if not core.is_floating_point():
            raise TypeError(f"core {index} must be floating point")
    if cores and len({core.dtype for core in cores}) != 1:
        raise ValueError("all TT cores must have the same dtype")
    if cores and len({core.device for core in cores}) != 1:
        raise ValueError("all TT cores must be on the same device")


def tensorize_matrix(weight: Tensor, spec: TTMatrixSpec) -> Tensor:
    expected = (spec.out_features, spec.in_features)
    if tuple(weight.shape) != expected:
        raise ValueError(f"expected matrix shape {expected}, got {tuple(weight.shape)}")
    order = spec.order
    shaped = weight.reshape(*spec.out_modes, *spec.in_modes)
    permutation = tuple(
        axis
        for pair in zip(range(order), range(order, 2 * order), strict=True)
        for axis in pair
    )
    return shaped.permute(permutation).contiguous().reshape(*spec.physical_modes)


def detensorize_matrix(tensor: Tensor, spec: TTMatrixSpec) -> Tensor:
    if tuple(tensor.shape) != spec.physical_modes:
        raise ValueError(
            f"expected tensor shape {spec.physical_modes}, got {tuple(tensor.shape)}"
        )
    shape = tuple(
        v for pair in zip(spec.out_modes, spec.in_modes, strict=True) for v in pair
    )
    permutation = tuple(range(0, 2 * spec.order, 2)) + tuple(
        range(1, 2 * spec.order, 2)
    )
    return (
        tensor.reshape(*shape)
        .permute(permutation)
        .contiguous()
        .reshape(spec.out_features, spec.in_features)
    )


def reconstruct_matrix(cores: Sequence[Tensor], spec: TTMatrixSpec) -> Tensor:
    validate_cores(cores, spec)
    state = cores[0]
    for core in cores[1:]:
        state = torch.tensordot(state, core, dims=([-1], [0]))
    return detensorize_matrix(
        state.squeeze(0).squeeze(-1).reshape(*spec.physical_modes), spec
    )


def slice_bond(
    cores: Sequence[Tensor], spec: TTMatrixSpec, *, bond_index: int, rank: int
) -> tuple[list[Tensor], TTMatrixSpec]:
    validate_cores(cores, spec)
    if not 1 <= bond_index < spec.order:
        raise ValueError(f"bond_index must be in [1, {spec.order - 1}]")
    if not 1 <= rank <= spec.ranks[bond_index]:
        raise ValueError(f"rank must be in [1, {spec.ranks[bond_index]}]")
    result = [core.clone() for core in cores]
    result[bond_index - 1] = result[bond_index - 1][..., :rank].contiguous()
    result[bond_index] = result[bond_index][:rank, ...].contiguous()
    result_spec = spec.with_bond_rank(bond_index, rank)
    validate_cores(result, result_spec)
    return result, result_spec
