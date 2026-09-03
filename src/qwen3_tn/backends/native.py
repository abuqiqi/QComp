"""Native PyTorch TT contraction runtime."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor
from ..tt import TTMatrixSpec, reconstruct_matrix
from .base import (
    ParameterListTTLinear,
    TTBackend,
    TTLinearBase,
    reject_backend_options,
)


def _forward_chunk(
    flat_input: Tensor, cores: Sequence[Tensor], spec: TTMatrixSpec
) -> Tensor:
    tokens = flat_input.shape[0]
    state = flat_input.reshape(tokens, *spec.in_modes)
    state = torch.tensordot(state, cores[-1].squeeze(-1), dims=([-1], [2]))
    for index in range(spec.order - 2, -1, -1):
        remaining = index + 1
        state = torch.tensordot(
            state, cores[index], dims=([remaining, remaining + 1], [2, 3])
        )
        accumulated = spec.order - index - 1
        raw_start = remaining
        rank_axis = raw_start + accumulated
        mode_axis = rank_axis + 1
        permutation = (
            [0]
            + list(range(1, remaining))
            + [rank_axis, mode_axis]
            + list(range(raw_start, raw_start + accumulated))
        )
        state = state.permute(permutation).contiguous()
    return state.squeeze(1).reshape(tokens, spec.out_features)


def native_tt_forward_flat(
    flat_input: Tensor,
    cores: Sequence[Tensor],
    spec: TTMatrixSpec,
    token_chunk_size: int,
) -> Tensor:
    """Contract canonical TT cores with a flat token matrix using PyTorch."""

    return torch.cat(
        [
            _forward_chunk(
                flat_input[start : start + token_chunk_size], cores, spec
            )
            for start in range(0, flat_input.shape[0], token_chunk_size)
        ],
        dim=0,
    )


class NativeTTLinear(ParameterListTTLinear):
    def __init__(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int = 8,
        trainable: bool = False,
        preserve_input_dtype: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )

    @property
    def backend_name(self) -> str:
        return "native"

    @property
    def backend_version(self) -> str:
        return "2"

    def reconstruct_weight(self) -> Tensor:
        return reconstruct_matrix(tuple(self.cores), self.spec)

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        output = native_tt_forward_flat(
            flat, self.cores, self.spec, self.token_chunk_size
        )
        return output.reshape(*shape, self.out_features)


class NativeTTBackend(TTBackend):
    name = "native"

    @property
    def backend_version(self) -> str:
        return "2"

    def normalize_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return reject_backend_options(self.name, options)

    def build_linear(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int,
        trainable: bool,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
        backend_options: Mapping[str, Any],
    ) -> TTLinearBase:
        self.normalize_options(backend_options)
        return NativeTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
