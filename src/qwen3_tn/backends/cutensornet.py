"""Optional cuTensorNet inference runtime with native decode fallback."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..tt import TTMatrixSpec
from .base import (
    ParameterListTTLinear,
    TTBackend,
    TTBackendCapabilities,
    TTBackendProbe,
    TTLinearBase,
)
from .native import native_tt_forward_flat

_INSTALL_HINT = 'python -m pip install cuquantum-python-cu12'


def _types() -> tuple[type[Any], type[Any]]:
    try:
        from cuquantum.tensornet import Network, NetworkOptions
    except ImportError as error:
        raise ImportError(
            f"cutensornet backend is optional; install it with `{_INSTALL_HINT}`"
        ) from error
    return Network, NetworkOptions


def _version() -> str:
    for distribution in ("cuquantum-python-cu12", "cuquantum-python"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            pass
    return "unknown"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class CuTensorNetOptions:
    """Validated configuration for the cuTensorNet runtime."""

    min_tokens: int = 16
    token_bucket_size: int = 128
    max_cached_networks: int = 2
    memory_limit: int | str = "256MiB"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CuTensorNetOptions":
        options = dict(values)
        known = {
            "min_tokens",
            "token_bucket_size",
            "max_cached_networks",
            "memory_limit",
        }
        unknown = set(options) - known
        if unknown:
            raise ValueError(
                f"cutensornet backend does not accept options: {sorted(unknown)}"
            )
        parsed = cls(**options)
        for name, value, allow_zero in (
            ("min_tokens", parsed.min_tokens, False),
            ("token_bucket_size", parsed.token_bucket_size, True),
            ("max_cached_networks", parsed.max_cached_networks, False),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"cutensornet {name} must be an integer")
            if value < (0 if allow_zero else 1):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"cutensornet {name} must be {qualifier}")
        if isinstance(parsed.memory_limit, bool) or not isinstance(
            parsed.memory_limit, (int, str)
        ):
            raise TypeError("cutensornet memory_limit must be an integer or string")
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_tokens": self.min_tokens,
            "token_bucket_size": self.token_bucket_size,
            "max_cached_networks": self.max_cached_networks,
            "memory_limit": self.memory_limit,
        }


class CuTensorNetTTLinear(ParameterListTTLinear):
    """Use planned cuTensorNet contractions for prefill-sized inputs."""

    def __init__(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int = 8,
        preserve_input_dtype: bool = True,
        activation_checkpointing: bool = False,
        options: CuTensorNetOptions | None = None,
    ) -> None:
        super().__init__(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=False,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
        self.options = options or CuTensorNetOptions()
        self._network_cache: OrderedDict[
            tuple[int, str, torch.dtype],
            tuple[Any, Tensor],
        ] = OrderedDict()

    @property
    def backend_name(self) -> str:
        return "cutensornet"

    @property
    def backend_version(self) -> str:
        return _version()

    def backend_metadata(self) -> dict[str, Any]:
        return {
            **super().backend_metadata(),
            "implementation": "cuquantum.tensornet.Network",
            "inference_only": True,
            **self.options.to_dict(),
        }

    def _free_networks(self) -> None:
        cache = getattr(self, "_network_cache", None)
        if cache is None:
            return
        while cache:
            _, (network, _) = cache.popitem(last=False)
            network.free()

    def close(self) -> None:
        """Release cached cuTensorNet networks deterministically."""

        self._free_networks()

    def _apply(self, fn: Any, recurse: bool = True) -> "CuTensorNetTTLinear":
        self._free_networks()
        return super()._apply(fn, recurse)

    def _after_cores_loaded(self) -> None:
        self._free_networks()

    def _bucket(self, tokens: int) -> int:
        if self.options.token_bucket_size == 0:
            return tokens
        return (
            (tokens + self.options.token_bucket_size - 1)
            // self.options.token_bucket_size
        ) * self.options.token_bucket_size

    def _network_operands(self, inputs: Tensor) -> list[Any]:
        order = self.spec.order
        input_modes = list(range(1, order + 1))
        output_modes = list(range(order + 1, 2 * order + 1))
        rank_modes = list(range(2 * order + 1, 3 * order + 2))
        operands: list[Any] = [inputs, [0, *input_modes]]
        for index, core in enumerate(self.cores):
            operands.extend(
                (
                    core,
                    [
                        rank_modes[index],
                        output_modes[index],
                        input_modes[index],
                        rank_modes[index + 1],
                    ],
                )
            )
        operands.append([0, *output_modes])
        return operands

    def _create_network(self, buffer: Tensor) -> Any:
        Network, NetworkOptions = _types()
        network = Network(
            *self._network_operands(buffer),
            options=NetworkOptions(
                memory_limit=self.options.memory_limit,
                blocking="auto",
            ),
        )
        try:
            network.contract_path()
        except Exception as error:
            network.free()
            message = str(error)
            if "cublasSetEnvironmentMode" in message:
                raise RuntimeError(
                    "cuTensorNet could not load a compatible cuBLAS. Start the "
                    "process with CUDA Toolkit cuBLAS and cuBLASLt in LD_PRELOAD."
                ) from error
            raise
        return network

    def _cutensornet_forward(self, flat: Tensor) -> Tensor:
        if flat.device.type != "cuda":
            raise RuntimeError("cutensornet backend requires CUDA inputs")
        tokens = flat.shape[0]
        bucket_tokens = self._bucket(tokens)
        key = (bucket_tokens, str(flat.device), flat.dtype)
        cached = self._network_cache.get(key)
        if cached is None:
            buffer = torch.empty(
                (bucket_tokens, *self.spec.in_modes),
                device=flat.device,
                dtype=flat.dtype,
            )
            network = self._create_network(buffer)
            self._network_cache[key] = (network, buffer)
            while len(self._network_cache) > self.options.max_cached_networks:
                _, (old_network, _) = self._network_cache.popitem(last=False)
                old_network.free()
        else:
            network, buffer = cached
            self._network_cache.move_to_end(key)
        tensorized = flat.reshape(tokens, *self.spec.in_modes)
        buffer[:tokens].copy_(tensorized)
        if bucket_tokens != tokens:
            buffer[tokens:].zero_()
        output = network.contract()
        return output[:tokens].reshape(tokens, self.out_features)

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        if torch.is_grad_enabled() and (
            flat.requires_grad or any(core.requires_grad for core in self.cores)
        ):
            raise RuntimeError(
                "cutensornet backend is inference-only and does not support backward"
            )
        if flat.shape[0] < self.options.min_tokens:
            output = native_tt_forward_flat(
                flat, self.cores, self.spec, self.token_chunk_size
            )
        else:
            output = self._cutensornet_forward(flat)
        return output.reshape(*shape, self.out_features)


class CuTensorNetBackend(TTBackend):
    name = "cutensornet"

    @property
    def backend_version(self) -> str:
        return _version()

    @property
    def capabilities(self) -> TTBackendCapabilities:
        return TTBackendCapabilities(
            supports_training=False,
            supports_backward=False,
            requires_cuda=True,
            supports_activation_checkpointing=False,
        )

    def probe(self) -> TTBackendProbe:
        if not _module_available("cuquantum.tensornet"):
            return TTBackendProbe(
                available=False, reason=f"install with `{_INSTALL_HINT}`"
            )
        if not torch.cuda.is_available():
            return TTBackendProbe(available=False, reason="CUDA is not available")
        return TTBackendProbe(available=True)

    def normalize_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return CuTensorNetOptions.from_mapping(options).to_dict()

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
        self.validate_request(
            trainable=trainable,
            activation_checkpointing=activation_checkpointing,
        )
        options = CuTensorNetOptions.from_mapping(backend_options)
        return CuTensorNetTTLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
            options=options,
        )
