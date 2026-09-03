"""Optional torchTT LinearLayerTT runtime."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..tt import TTMatrixSpec, validate_cores
from .base import (
    TTBackend,
    TTBackendProbe,
    TTLinearBase,
    reject_backend_options,
)

_INSTALL_HINT = 'pip install -e ".[torchtt]"'


def _linear_type() -> type[nn.Module]:
    try:
        from torchtt.nn import LinearLayerTT
    except ImportError as error:
        raise ImportError(
            f"torchtt backend is optional; install it with `{_INSTALL_HINT}`"
        ) from error
    return LinearLayerTT


def _version() -> str:
    try:
        return version("torchTT")
    except PackageNotFoundError:
        return "unknown"


class TorchTTLinear(TTLinearBase):
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
        LinearLayerTT = _linear_type()
        linear = LinearLayerTT(
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
        self._configure_runtime(
            spec,
            token_chunk_size=token_chunk_size,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
        self.linear = linear

    @property
    def backend_name(self) -> str:
        return "torchtt"

    @property
    def backend_version(self) -> str:
        return _version()

    def backend_metadata(self) -> dict[str, Any]:
        return {
            **super().backend_metadata(),
            "implementation": "LinearLayerTT",
        }

    def tt_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.linear.cores)

    def _forward_chunk(self, flat: Tensor) -> Tensor:
        tensorized = flat.reshape(flat.shape[0], *self.spec.in_modes)
        return self.linear(tensorized).reshape(flat.shape[0], self.out_features)

    def _forward_flat(self, flat: Tensor) -> Tensor:
        return torch.cat(
            [
                self._forward_chunk(flat[start : start + self.token_chunk_size])
                for start in range(0, flat.shape[0], self.token_chunk_size)
            ],
            dim=0,
        )

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        return self._forward_flat(flat).reshape(*shape, self.out_features)


class TorchTTBackend(TTBackend):
    name = "torchtt"

    @property
    def backend_version(self) -> str:
        return _version()

    def probe(self) -> TTBackendProbe:
        available = importlib.util.find_spec("torchtt") is not None
        return TTBackendProbe(
            available=available,
            reason=None if available else f"install with `{_INSTALL_HINT}`",
        )

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
        return TorchTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
