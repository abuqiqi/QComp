"""Backend contracts and shared runtime behavior for TT linear layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..tt import TTMatrixSpec, validate_cores


@dataclass(frozen=True)
class TTBackendCapabilities:
    """Operations and runtime requirements supported by a backend."""

    supports_training: bool = True
    supports_backward: bool = True
    requires_cuda: bool = False
    supports_activation_checkpointing: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TTBackendProbe:
    """Result of checking whether a registered backend can run here."""

    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TTLinearBase(nn.Module, ABC):
    """Common TT layer lifecycle; subclasses only provide the computation."""

    spec: TTMatrixSpec
    token_chunk_size: int
    preserve_input_dtype: bool
    activation_checkpointing: bool

    def _configure_runtime(
        self,
        spec: TTMatrixSpec,
        *,
        token_chunk_size: int,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
    ) -> None:
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.spec = spec
        self.token_chunk_size = int(token_chunk_size)
        self.preserve_input_dtype = bool(preserve_input_dtype)
        self.activation_checkpointing = bool(activation_checkpointing)

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def backend_version(self) -> str: ...

    @abstractmethod
    def tt_parameters(self) -> tuple[nn.Parameter, ...]: ...

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_impl() or forward()"
        )

    @property
    def in_features(self) -> int:
        return self.spec.in_features

    @property
    def out_features(self) -> int:
        return self.spec.out_features

    def export_cores(self) -> list[Tensor]:
        cores = [parameter.detach() for parameter in self.tt_parameters()]
        validate_cores(cores, self.spec)
        return cores

    def load_cores(self, cores: Sequence[Tensor]) -> None:
        validate_cores(cores, self.spec)
        parameters = self.tt_parameters()
        with torch.no_grad():
            for parameter, saved in zip(parameters, cores, strict=True):
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
        self._after_cores_loaded()

    def _after_cores_loaded(self) -> None:
        """Invalidate backend-specific state after core values change."""

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = bool(enabled)

    def backend_metadata(self) -> dict[str, Any]:
        return {"name": self.backend_name, "version": self.backend_version}

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last input dimension {self.in_features}, "
                f"got {inputs.shape[-1]}"
            )
        if inputs.numel() == 0:
            return inputs.new_empty(*inputs.shape[:-1], self.out_features)
        parameters = self.tt_parameters()
        if not parameters:
            raise RuntimeError(f"{self.backend_name} backend has no TT parameters")
        input_dtype = inputs.dtype
        compute_input = inputs.to(parameters[0].dtype)
        use_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and any(parameter.requires_grad for parameter in parameters)
        )
        output = (
            checkpoint(
                self._forward_impl,
                compute_input,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if use_checkpoint
            else self._forward_impl(compute_input)
        )
        expected_shape = (*inputs.shape[:-1], self.out_features)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{self.backend_name} backend returned shape {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        return (
            output.to(input_dtype)
            if self.preserve_input_dtype and output.dtype != input_dtype
            else output
        )


class ParameterListTTLinear(TTLinearBase):
    """Shared storage implementation for backends using canonical TT cores."""

    def __init__(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int,
        trainable: bool,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        validate_cores(cores, spec)
        self._configure_runtime(
            spec,
            token_chunk_size=token_chunk_size,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
        self.cores = nn.ParameterList(
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        )

    def tt_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.cores)


class TTBackend(ABC):
    name: str

    @property
    def backend_version(self) -> str:
        return "unknown"

    @property
    def capabilities(self) -> TTBackendCapabilities:
        return TTBackendCapabilities()

    def probe(self) -> TTBackendProbe:
        return TTBackendProbe(available=True)

    def normalize_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return dict(options)

    def validate_request(
        self, *, trainable: bool, activation_checkpointing: bool
    ) -> None:
        capabilities = self.capabilities
        if trainable and not capabilities.supports_training:
            raise RuntimeError(
                f"{self.name} backend is inference-only and does not support training"
            )
        if activation_checkpointing and not capabilities.supports_activation_checkpointing:
            raise RuntimeError(
                f"{self.name} backend does not support activation checkpointing"
            )

    def backend_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.backend_version}

    @abstractmethod
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
    ) -> TTLinearBase: ...


def reject_backend_options(
    backend_name: str, options: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = dict(options)
    if normalized:
        raise ValueError(
            f"{backend_name} backend does not accept options: {sorted(normalized)}"
        )
    return normalized
