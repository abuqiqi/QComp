"""TT-SVD for dense matrices."""

from __future__ import annotations
import torch
from torch import Tensor
from .operations import tensorize_matrix, validate_cores
from .spec import TTMatrixSpec


def _svd(matrix: Tensor, driver: str | None) -> tuple[Tensor, Tensor, Tensor]:
    kwargs: dict[str, object] = {"full_matrices": False}
    if matrix.is_cuda and driver is not None:
        kwargs["driver"] = driver
    try:
        return torch.linalg.svd(matrix, **kwargs)
    except RuntimeError:
        if not matrix.is_cuda or driver in (None, "gesvd"):
            raise
        return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")


def tt_svd_matrix(
    weight: Tensor, spec: TTMatrixSpec, *, svd_driver: str | None = "gesvdj"
) -> list[Tensor]:
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")
    remainder = tensorize_matrix(weight, spec)
    cores: list[Tensor] = []
    left_rank = 1
    for index in range(spec.order - 1):
        matrix = remainder.reshape(left_rank * spec.physical_modes[index], -1)
        target_rank = spec.ranks[index + 1]
        if target_rank > min(matrix.shape):
            raise ValueError(
                f"requested rank {target_rank} exceeds unfolding shape {tuple(matrix.shape)}"
            )
        u, singular_values, vh = _svd(matrix, svd_driver)
        cores.append(
            u[:, :target_rank]
            .reshape(
                left_rank, spec.out_modes[index], spec.in_modes[index], target_rank
            )
            .contiguous()
        )
        remainder = (
            singular_values[:target_rank].unsqueeze(1) * vh[:target_rank]
        ).reshape(target_rank, *spec.physical_modes[index + 1 :])
        left_rank = target_rank
    cores.append(
        remainder.reshape(
            left_rank, spec.out_modes[-1], spec.in_modes[-1], 1
        ).contiguous()
    )
    validate_cores(cores, spec)
    return cores
