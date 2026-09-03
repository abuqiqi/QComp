"""Lazy, explicitly extensible TT backend registry."""

from __future__ import annotations

import importlib
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from torch import Tensor

from ..tt import TTMatrixSpec
from .base import TTBackend, TTBackendProbe, TTLinearBase

TTBackendFactory = Callable[[], TTBackend]

_BACKEND_FACTORIES: dict[str, TTBackendFactory] = {}
_BACKEND_INSTANCES: dict[str, TTBackend] = {}
_REGISTRY_LOCK = RLock()


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("TT backend name must not be empty")
    return normalized


def register_tt_backend(
    name: str, factory: TTBackendFactory, *, replace: bool = False
) -> None:
    """Register a zero-argument backend factory under a stable name."""

    normalized = _normalize_name(name)
    if not callable(factory):
        raise TypeError("TT backend factory must be callable")
    with _REGISTRY_LOCK:
        if normalized in _BACKEND_FACTORIES and not replace:
            raise ValueError(f"TT backend is already registered: {normalized}")
        _BACKEND_FACTORIES[normalized] = factory
        _BACKEND_INSTANCES.pop(normalized, None)


def _lazy_factory(module_name: str, class_name: str) -> TTBackendFactory:
    def factory() -> TTBackend:
        module = importlib.import_module(module_name, package=__package__)
        backend_type = getattr(module, class_name)
        return backend_type()

    return factory


def _register_builtin_backends() -> None:
    builtins = {
        "native": (".native", "NativeTTBackend"),
        "tensorly_torch": (".tensorly_torch", "TensorLyTorchTTBackend"),
        "torchtt": (".torchtt", "TorchTTBackend"),
        "cutensornet": (".cutensornet", "CuTensorNetBackend"),
    }
    for name, (module_name, class_name) in builtins.items():
        register_tt_backend(name, _lazy_factory(module_name, class_name))


_register_builtin_backends()


def import_tt_backend_modules(modules: Sequence[str] | None) -> tuple[str, ...]:
    """Import trusted modules whose import side effects register TT backends."""

    imported: list[str] = []
    for module in modules or ():
        name = str(module).strip()
        if not name:
            raise ValueError("backend module name must not be empty")
        importlib.import_module(name)
        imported.append(name)
    return tuple(imported)


def registered_tt_backends() -> tuple[str, ...]:
    """Return all backend names known to the registry."""

    with _REGISTRY_LOCK:
        return tuple(_BACKEND_FACTORIES)


def get_tt_backend(name: str) -> TTBackend:
    normalized = _normalize_name(name)
    with _REGISTRY_LOCK:
        try:
            factory = _BACKEND_FACTORIES[normalized]
        except KeyError as error:
            raise ValueError(
                f"unknown TT backend {name!r}; "
                f"registered={list(registered_tt_backends())!r}"
            ) from error
        if normalized not in _BACKEND_INSTANCES:
            backend = factory()
            if not isinstance(backend, TTBackend):
                raise TypeError(
                    f"TT backend factory {normalized!r} returned "
                    f"{type(backend).__name__}, expected TTBackend"
                )
            if _normalize_name(backend.name) != normalized:
                raise ValueError(
                    f"TT backend factory {normalized!r} returned backend named "
                    f"{backend.name!r}"
                )
            _BACKEND_INSTANCES[normalized] = backend
        return _BACKEND_INSTANCES[normalized]


def probe_tt_backend(name: str) -> TTBackendProbe:
    """Check optional dependencies and runtime requirements without building a layer."""

    probe = get_tt_backend(name).probe()
    if not isinstance(probe, TTBackendProbe):
        raise TypeError(
            f"TT backend {name!r} returned {type(probe).__name__}, "
            "expected TTBackendProbe"
        )
    return probe


def available_tt_backends() -> tuple[str, ...]:
    """Return registered backends usable in the current environment."""

    return tuple(
        name for name in registered_tt_backends() if probe_tt_backend(name).available
    )


def tt_backend_metadata(
    name: str, backend_options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    backend = get_tt_backend(name)
    options = backend.normalize_options(dict(backend_options or {}))
    return {
        **backend.backend_metadata(),
        "options": options,
    }


def create_tt_linear(
    spec: TTMatrixSpec,
    cores: Sequence[Tensor],
    *,
    tt_backend: str = "native",
    token_chunk_size: int = 8,
    trainable: bool = False,
    preserve_input_dtype: bool = True,
    activation_checkpointing: bool = False,
    backend_options: Mapping[str, Any] | None = None,
) -> TTLinearBase:
    backend = get_tt_backend(tt_backend)
    options = backend.normalize_options(dict(backend_options or {}))
    backend.validate_request(
        trainable=trainable,
        activation_checkpointing=activation_checkpointing,
    )
    layer = backend.build_linear(
        spec,
        cores,
        token_chunk_size=token_chunk_size,
        trainable=trainable,
        preserve_input_dtype=preserve_input_dtype,
        activation_checkpointing=activation_checkpointing,
        backend_options=options,
    )
    if not isinstance(layer, TTLinearBase):
        raise TypeError(
            f"TT backend {backend.name!r} returned {type(layer).__name__}, "
            "expected TTLinearBase"
        )
    if (
        _normalize_name(layer.backend_name) != _normalize_name(backend.name)
        or layer.backend_version != backend.backend_version
    ):
        raise ValueError(
            f"TT backend identity mismatch: factory={backend.backend_metadata()!r}, "
            f"layer={layer.backend_metadata()!r}"
        )
    return layer
