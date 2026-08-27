"""Native PyTorch TT contraction runtime."""

from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from ..tt import TTMatrixSpec, reconstruct_matrix, validate_cores
from .base import TTBackend, TTLinearBase


class NativeTTLinear(TTLinearBase):
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
        super().__init__()
        validate_cores(cores, spec)
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.spec = spec
        self.token_chunk_size = int(token_chunk_size)
        self.preserve_input_dtype = bool(preserve_input_dtype)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.cores = nn.ParameterList(
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        )

    @property
    def backend_name(self) -> str:
        return "native"

    @property
    def backend_version(self) -> str:
        return "2"

    def tt_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.cores)

    def export_cores(self) -> list[Tensor]:
        return [core.detach() for core in self.cores]

    def load_cores(self, cores: Sequence[Tensor]) -> None:
        validate_cores(cores, self.spec)
        with torch.no_grad():
            for parameter, saved in zip(self.cores, cores, strict=True):
                parameter.copy_(saved.to(parameter))

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = bool(enabled)

    def reconstruct_weight(self) -> Tensor:
        return reconstruct_matrix(tuple(self.cores), self.spec)

    def _forward_chunk(self, flat_input: Tensor) -> Tensor:
        tokens = flat_input.shape[0]
        state = flat_input.reshape(tokens, *self.spec.in_modes)
        state = torch.tensordot(state, self.cores[-1].squeeze(-1), dims=([-1], [2]))
        for index in range(self.spec.order - 2, -1, -1):
            remaining = index + 1
            state = torch.tensordot(
                state, self.cores[index], dims=([remaining, remaining + 1], [2, 3])
            )
            accumulated = self.spec.order - index - 1
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
        return state.squeeze(1).reshape(tokens, self.out_features)

    def _forward_flat(self, flat: Tensor) -> Tensor:
        return torch.cat(
            [
                self._forward_chunk(flat[start : start + self.token_chunk_size])
                for start in range(0, flat.shape[0], self.token_chunk_size)
            ],
            dim=0,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last input dimension {self.in_features}, got {inputs.shape[-1]}"
            )
        if inputs.numel() == 0:
            return inputs.new_empty(*inputs.shape[:-1], self.out_features)
        shape = inputs.shape[:-1]
        input_dtype = inputs.dtype
        flat = inputs.reshape(-1, self.in_features).to(self.cores[0].dtype)
        use_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and any(core.requires_grad for core in self.cores)
        )
        output = (
            checkpoint(
                self._forward_flat, flat, use_reentrant=False, preserve_rng_state=False
            )
            if use_checkpoint
            else self._forward_flat(flat)
        )
        output = output.reshape(*shape, self.out_features)
        return (
            output.to(input_dtype)
            if self.preserve_input_dtype and output.dtype != input_dtype
            else output
        )


class NativeTTBackend(TTBackend):
    name = "native"

    def build_linear(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int,
        trainable: bool,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
    ) -> TTLinearBase:
        return NativeTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
