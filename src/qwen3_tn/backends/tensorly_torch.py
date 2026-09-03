"""Optional TensorLy-Torch BlockTT runtime."""

from __future__ import annotations
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Sequence
from torch import Tensor, nn
from ..tt import TTMatrixSpec, validate_cores
from .base import (
    TTBackend,
    TTBackendProbe,
    TTLinearBase,
    reject_backend_options,
)

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
        self._configure_runtime(
            spec,
            token_chunk_size=token_chunk_size,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
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

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)


class TensorLyTorchTTBackend(TTBackend):
    name = "tensorly_torch"

    @property
    def backend_version(self) -> str:
        return _version()

    def probe(self) -> TTBackendProbe:
        available = importlib.util.find_spec("tltorch") is not None
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
        return TensorLyTorchTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
