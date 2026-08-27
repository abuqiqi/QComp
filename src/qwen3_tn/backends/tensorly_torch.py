"""Optional TensorLy-Torch BlockTT runtime."""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from ..tt import TTMatrixSpec, validate_cores
from .base import TTBackend, TTLinearBase

_INSTALL_HINT = 'pip install -e ".[tensorly]"'


def _types() -> tuple[type[Any], type[Any]]:
    try:
        from tltorch import BlockTT, FactorizedLinear
    except ImportError as error:
        raise ImportError(
            f"tensorly_torch backend is optional; install it with `{_INSTALL_HINT}`"
        ) from error
    return BlockTT, FactorizedLinear


def _version() -> str:
    try:
        return version("tensorly-torch")
    except PackageNotFoundError:
        return "unknown"


class TensorLyTorchTTLinear(TTLinearBase):
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
        BlockTT, FactorizedLinear = _types()
        parameters = [
            nn.Parameter(core.detach().clone(), requires_grad=trainable)
            for core in cores
        ]
        block_tt = BlockTT(
            parameters,
            tensorized_shape=(spec.out_modes, spec.in_modes),
            rank=spec.ranks,
        )
        self.spec = spec
        self.token_chunk_size = int(token_chunk_size)
        self.preserve_input_dtype = bool(preserve_input_dtype)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.linear = FactorizedLinear(
            in_tensorized_features=spec.in_modes,
            out_tensorized_features=spec.out_modes,
            bias=False,
            factorization=block_tt,
            implementation="factorized",
            checkpointing=False,
            device=cores[0].device,
            dtype=cores[0].dtype,
        )

    @property
    def backend_name(self) -> str:
        return "tensorly_torch"

    @property
    def backend_version(self) -> str:
        return _version()

    def backend_metadata(self) -> dict[str, Any]:
        return {
            **super().backend_metadata(),
            "factorization": "BlockTT",
            "implementation": "factorized",
        }

    def tt_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.linear.weight.factors)

    def export_cores(self) -> list[Tensor]:
        cores = [p.detach() for p in self.tt_parameters()]
        validate_cores(cores, self.spec)
        return cores

    def load_cores(self, cores: Sequence[Tensor]) -> None:
        validate_cores(cores, self.spec)
        with torch.no_grad():
            for parameter, saved in zip(self.tt_parameters(), cores, strict=True):
                parameter.copy_(saved.to(parameter))

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = bool(enabled)

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last input dimension {self.in_features}, got {inputs.shape[-1]}"
            )
        if inputs.numel() == 0:
            return inputs.new_empty(*inputs.shape[:-1], self.out_features)
        input_dtype = inputs.dtype
        compute_dtype = self.tt_parameters()[0].dtype
        compute_input = inputs.to(compute_dtype)
        use_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.tt_parameters())
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
        return (
            output.to(input_dtype)
            if self.preserve_input_dtype and output.dtype != input_dtype
            else output
        )


class TensorLyTorchTTBackend(TTBackend):
    name = "tensorly_torch"

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
        return TensorLyTorchTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
