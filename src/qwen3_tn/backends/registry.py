"""Lazy backend registry."""

from __future__ import annotations
from functools import lru_cache
from typing import Sequence
from torch import Tensor
from ..tt import TTMatrixSpec
from .base import TTBackend, TTLinearBase

_SUPPORTED = ("native", "tensorly_torch")


def available_tt_backends() -> tuple[str, ...]:
    return _SUPPORTED


@lru_cache(maxsize=None)
def get_tt_backend(name: str) -> TTBackend:
    normalized = str(name).strip().lower()
    if normalized == "native":
        from .native import NativeTTBackend

        return NativeTTBackend()
    if normalized == "tensorly_torch":
        from .tensorly_torch import TensorLyTorchTTBackend

        return TensorLyTorchTTBackend()
    raise ValueError(f"unknown TT backend {name!r}; available={list(_SUPPORTED)!r}")


def create_tt_linear(
    spec: TTMatrixSpec,
    cores: Sequence[Tensor],
    *,
    tt_backend: str = "native",
    token_chunk_size: int = 8,
    trainable: bool = False,
    preserve_input_dtype: bool = True,
    activation_checkpointing: bool = False,
) -> TTLinearBase:
    return get_tt_backend(tt_backend).build_linear(
        spec,
        cores,
        token_chunk_size=token_chunk_size,
        trainable=trainable,
        preserve_input_dtype=preserve_input_dtype,
        activation_checkpointing=activation_checkpointing,
    )
