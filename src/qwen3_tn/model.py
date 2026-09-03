"""Architecture-neutral installation and export of TT modules."""

from __future__ import annotations
import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import torch
from torch import nn
from .backends import TTLinearBase, create_tt_linear
from .checkpoint import (
    TTModuleSetEntry,
    load_module_set_index,
    load_tt_module,
    publish_module_set,
    save_module_set_index,
    save_tt_module,
)
from .tt import TTMatrixSpec


@dataclass(frozen=True)
class TTTarget:
    module_path: str
    spec: TTMatrixSpec
    token_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not self.module_path:
            raise ValueError("module_path must not be empty")
        if self.token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TTTarget":
        return cls(
            str(value["module_path"]),
            TTMatrixSpec.from_dict(value),
            int(value.get("token_chunk_size", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            **self.spec.to_dict(),
            "token_chunk_size": self.token_chunk_size,
        }


def _parent_and_name(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parent_path, separator, name = module_path.rpartition(".")
    if not separator:
        return model, module_path
    try:
        parent = model.get_submodule(parent_path)
    except AttributeError as error:
        raise KeyError(f"module path does not exist: {module_path}") from error
    return parent, name


def _get_module(model: nn.Module, module_path: str) -> nn.Module:
    try:
        return model.get_submodule(module_path)
    except AttributeError as error:
        raise KeyError(f"module path does not exist: {module_path}") from error


def _replace_module(model: nn.Module, module_path: str, module: nn.Module) -> None:
    parent, name = _parent_and_name(model, module_path)
    if name not in parent._modules:
        raise KeyError(f"module path does not exist: {module_path}")
    parent._modules[name] = module


def _install_replacements(
    model: nn.Module,
    originals: Mapping[str, nn.Module],
    replacements: Mapping[str, nn.Module],
) -> None:
    installed: list[str] = []
    try:
        for path, replacement in replacements.items():
            _replace_module(model, path, replacement)
            installed.append(path)
    except Exception:
        for path in reversed(installed):
            _replace_module(model, path, originals[path])
        raise


def find_tt_modules(model: nn.Module) -> dict[str, TTLinearBase]:
    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, TTLinearBase)
    }


def _select_entries(
    entries: Sequence[Any], module_paths: Iterable[str] | None
) -> list[Any]:
    selected = list(entries)
    if module_paths is not None:
        requested = set(module_paths)
        available = {entry.module_path for entry in selected}
        missing = requested - available
        if missing:
            raise KeyError(
                f"module paths are absent from module-set index: {sorted(missing)}"
            )
        selected = [entry for entry in selected if entry.module_path in requested]
    if not selected:
        raise ValueError("no TT modules selected")
    return selected


def _prepare(
    model: nn.Module,
    index: str | Path,
    *,
    tt_backend: str,
    trainable: bool,
    core_dtype: torch.dtype | None,
    token_chunk_size: int | None,
    module_paths: Iterable[str] | None,
    backend_options: Mapping[str, Any] | None,
) -> tuple[dict[str, nn.Linear], dict[str, TTLinearBase]]:
    root, module_set = load_module_set_index(index, verify_modules=False)
    entries = _select_entries(module_set.modules, module_paths)
    originals: dict[str, nn.Linear] = {}
    replacements: dict[str, TTLinearBase] = {}
    for entry in entries:
        current = _get_module(model, entry.module_path)
        if not isinstance(current, nn.Linear):
            raise TypeError(
                f"{entry.module_path} must be bias-free torch.nn.Linear, got {type(current).__name__}"
            )
        if current.bias is not None:
            raise ValueError(f"{entry.module_path} must not have a bias")
        cores, manifest = load_tt_module(root / entry.checkpoint)
        expected = (manifest.spec.out_features, manifest.spec.in_features)
        if tuple(current.weight.shape) != expected:
            raise ValueError(
                f"{entry.module_path} weight shape {tuple(current.weight.shape)} does not match TT shape {expected}"
            )
        dtype = core_dtype or cores[0].dtype
        cores = [core.to(device=current.weight.device, dtype=dtype) for core in cores]
        chunk = (
            token_chunk_size
            if token_chunk_size is not None
            else int(manifest.metadata.get("token_chunk_size", 8))
        )
        originals[entry.module_path] = current
        replacements[entry.module_path] = create_tt_linear(
            manifest.spec,
            cores,
            tt_backend=tt_backend,
            token_chunk_size=chunk,
            trainable=trainable,
            preserve_input_dtype=True,
            backend_options=backend_options,
        )
    return originals, replacements


def install_tt_modules(
    model: nn.Module,
    index: str | Path,
    *,
    tt_backend: str = "native",
    trainable: bool = False,
    core_dtype: torch.dtype | None = None,
    token_chunk_size: int | None = None,
    module_paths: Iterable[str] | None = None,
    backend_options: Mapping[str, Any] | None = None,
) -> dict[str, TTLinearBase]:
    originals, replacements = _prepare(
        model,
        index,
        tt_backend=tt_backend,
        trainable=trainable,
        core_dtype=core_dtype,
        token_chunk_size=token_chunk_size,
        module_paths=module_paths,
        backend_options=backend_options,
    )
    _install_replacements(model, originals, replacements)
    return replacements


def _module_directory_name(module_path: str) -> str:
    readable = module_path.replace(".", "__")
    digest = hashlib.sha256(module_path.encode()).hexdigest()[:10]
    return f"{readable}--{digest}"


def export_tt_modules(
    model: nn.Module,
    output_root: str | Path,
    *,
    model_path: str = "",
    purpose: str = "",
    core_dtype: torch.dtype | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    modules = find_tt_modules(model)
    if not modules:
        raise RuntimeError("model contains no TT modules")
    target = Path(output_root)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    entries: list[TTModuleSetEntry] = []
    try:
        for path, module in modules.items():
            relative = Path("modules") / _module_directory_name(path)
            cores = module.export_cores()
            cores = (
                [core.to(dtype=core_dtype) for core in cores]
                if core_dtype is not None
                else cores
            )
            manifest = save_tt_module(
                temporary / relative,
                cores,
                module.spec,
                module_path=path,
                model_path=model_path,
                purpose=purpose,
                metadata={
                    "token_chunk_size": module.token_chunk_size,
                    "tt_backend": module.backend_metadata(),
                    **dict(metadata or {}),
                },
            )
            entries.append(
                TTModuleSetEntry(path, relative.as_posix(), str(manifest.cores_sha256))
            )
        save_module_set_index(
            temporary,
            entries,
            model_path=model_path,
            purpose=purpose,
            metadata=metadata,
        )
        load_module_set_index(temporary)
        publish_module_set(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "index.json"


def load_tt_cores(
    model: nn.Module,
    index: str | Path,
    *,
    module_paths: Iterable[str] | None = None,
    require_same_backend: bool = False,
) -> None:
    modules = find_tt_modules(model)
    root, module_set = load_module_set_index(index, verify_modules=False)
    entries = _select_entries(module_set.modules, module_paths)
    for entry in entries:
        if entry.module_path not in modules:
            raise KeyError(f"installed TT module not found: {entry.module_path}")
        cores, manifest = load_tt_module(root / entry.checkpoint)
        module = modules[entry.module_path]
        if manifest.spec != module.spec:
            raise ValueError(f"TT spec mismatch for {entry.module_path}")
        saved_backend = (
            manifest.metadata.get("tt_backend", {}).get("name")
            if isinstance(manifest.metadata.get("tt_backend"), Mapping)
            else None
        )
        if (
            require_same_backend
            and saved_backend
            and saved_backend != module.backend_name
        ):
            raise ValueError(
                f"backend mismatch for {entry.module_path}: checkpoint={saved_backend}, installed={module.backend_name}"
            )
        module.load_cores(cores)


class TTModulePatch:
    """Temporarily replace dense layers and always restore them on exit."""

    def __init__(
        self, model: nn.Module, index: str | Path, **install_options: Any
    ) -> None:
        self.model = model
        self.index = index
        self.install_options = install_options
        self.originals: dict[str, nn.Linear] = {}
        self.modules: dict[str, TTLinearBase] = {}

    def __enter__(self) -> dict[str, TTLinearBase]:
        if self.modules:
            raise RuntimeError("TTModulePatch is already active")
        options = dict(self.install_options)
        self.originals, self.modules = _prepare(
            self.model,
            self.index,
            tt_backend=options.pop("tt_backend", "native"),
            trainable=options.pop("trainable", False),
            core_dtype=options.pop("core_dtype", None),
            token_chunk_size=options.pop("token_chunk_size", None),
            module_paths=options.pop("module_paths", None),
            backend_options=options.pop("backend_options", None),
        )
        if options:
            raise TypeError(f"unknown install options: {sorted(options)}")
        try:
            _install_replacements(self.model, self.originals, self.modules)
        except Exception:
            self.modules = {}
            self.originals = {}
            raise
        return self.modules

    def __exit__(self, *_: Any) -> None:
        for path, original in reversed(tuple(self.originals.items())):
            _replace_module(self.model, path, original)
        self.modules = {}
        self.originals = {}
