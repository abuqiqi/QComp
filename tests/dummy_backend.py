"""Explicitly registered third-party backend used by integration tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor

from qwen3_tn import register_tt_backend
from qwen3_tn.backends import TTBackend, TTLinearBase
from qwen3_tn.backends.native import NativeTTLinear
from qwen3_tn.tt import TTMatrixSpec


class DummyTTLinear(NativeTTLinear):
    @property
    def backend_name(self) -> str:
        return "test_external"

    @property
    def backend_version(self) -> str:
        return "test-1"


class DummyTTBackend(TTBackend):
    name = "test_external"
    last_options: dict[str, Any] | None = None

    @property
    def backend_version(self) -> str:
        return "test-1"

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
        options = dict(backend_options)
        if set(options) - {"marker"}:
            raise ValueError(f"unknown dummy options: {sorted(options)}")
        self.last_options = options
        return DummyTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )


register_tt_backend("test_external", DummyTTBackend, replace=True)
