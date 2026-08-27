"""Backend contract for TT linear layers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
from torch import Tensor, nn
from ..tt import TTMatrixSpec


class TTLinearBase(nn.Module, ABC):
    spec: TTMatrixSpec
    token_chunk_size: int
    preserve_input_dtype: bool

    @property
    @abstractmethod
    def backend_name(self) -> str: ...
    @property
    @abstractmethod
    def backend_version(self) -> str: ...
    @abstractmethod
    def tt_parameters(self) -> tuple[nn.Parameter, ...]: ...
    @abstractmethod
    def export_cores(self) -> list[Tensor]: ...
    @abstractmethod
    def load_cores(self, cores: Sequence[Tensor]) -> None: ...
    @abstractmethod
    def set_activation_checkpointing(self, enabled: bool) -> None: ...

    @property
    def in_features(self) -> int:
        return self.spec.in_features

    @property
    def out_features(self) -> int:
        return self.spec.out_features

    def backend_metadata(self) -> dict[str, Any]:
        return {"name": self.backend_name, "version": self.backend_version}


class TTBackend(ABC):
    name: str

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
    ) -> TTLinearBase: ...
