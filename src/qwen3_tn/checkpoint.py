"""Canonical, backend-neutral TT module checkpoints."""

from __future__ import annotations
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from .provenance import atomic_write_json, file_sha256, load_json
from .tt import TTMatrixSpec, validate_cores

TT_MODULE_FORMAT = "qwen3-tn-tt-matrix-v1"
TT_MODULE_SET_FORMAT = "qwen3-tn-tt-module-set-v1"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class TTModuleManifest:
    module_path: str
    model_path: str
    purpose: str
    spec: TTMatrixSpec
    core_dtype: str
    checkpoint_bytes: int
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cores_sha256: str | None = None
    format: str = TT_MODULE_FORMAT

    def __post_init__(self) -> None:
        if self.format != TT_MODULE_FORMAT:
            raise ValueError(f"unsupported TT module format: {self.format}")
        if not self.module_path:
            raise ValueError("module_path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["spec"] = self.spec.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TTModuleManifest":
        try:
            return cls(
                module_path=str(value["module_path"]),
                model_path=str(value.get("model_path", "")),
                purpose=str(value.get("purpose", "")),
                spec=TTMatrixSpec.from_dict(value["spec"]),
                core_dtype=str(value["core_dtype"]),
                checkpoint_bytes=int(value["checkpoint_bytes"]),
                metrics=dict(value.get("metrics", {})),
                metadata=dict(value.get("metadata", {})),
                cores_sha256=value.get("cores_sha256"),
                format=str(value.get("format", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid TT module manifest") from error


@dataclass(frozen=True)
class TTModuleSetEntry:
    module_path: str
    checkpoint: str
    cores_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.checkpoint)
        if not self.module_path:
            raise ValueError("module_path must not be empty")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint must be a safe relative path")
        if len(self.cores_sha256) != 64:
            raise ValueError("cores_sha256 must be a SHA-256 hex digest")


@dataclass(frozen=True)
class TTModuleSetManifest:
    modules: tuple[TTModuleSetEntry, ...]
    model_path: str = ""
    purpose: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = TT_MODULE_SET_FORMAT

    def __post_init__(self) -> None:
        if self.format != TT_MODULE_SET_FORMAT:
            raise ValueError(f"unsupported TT module-set format: {self.format}")
        if not self.modules:
            raise ValueError("module set must not be empty")
        paths = [entry.module_path for entry in self.modules]
        if len(set(paths)) != len(paths):
            raise ValueError("module set contains duplicate module paths")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model_path": self.model_path,
            "purpose": self.purpose,
            "metadata": dict(self.metadata),
            "modules": [asdict(entry) for entry in self.modules],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TTModuleSetManifest":
        try:
            entries = tuple(TTModuleSetEntry(**entry) for entry in value["modules"])
            return cls(
                entries,
                str(value.get("model_path", "")),
                str(value.get("purpose", "")),
                dict(value.get("metadata", {})),
                str(value.get("format", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid TT module-set manifest") from error


def save_tt_module(
    directory: str | Path,
    cores: Sequence[Tensor],
    spec: TTMatrixSpec,
    *,
    module_path: str,
    model_path: str = "",
    purpose: str = "",
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TTModuleManifest:
    validate_cores(cores, spec)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f".cores.tmp-{uuid.uuid4().hex}.safetensors"
    weights = target / "cores.safetensors"
    save_file(
        {f"core_{i}": core.detach().cpu().contiguous() for i, core in enumerate(cores)},
        str(temporary),
    )
    os.replace(temporary, weights)
    manifest = TTModuleManifest(
        module_path,
        model_path,
        purpose,
        spec,
        _dtype_name(cores[0].dtype),
        weights.stat().st_size,
        dict(metrics or {}),
        dict(metadata or {}),
        file_sha256(weights),
    )
    atomic_write_json(target / "manifest.json", manifest.to_dict())
    return manifest


def load_tt_module(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_hash: bool = True,
) -> tuple[list[Tensor], TTModuleManifest]:
    source = Path(directory)
    manifest = TTModuleManifest.from_dict(load_json(source / "manifest.json"))
    weights = source / "cores.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"missing TT cores: {weights}")
    if weights.stat().st_size != manifest.checkpoint_bytes:
        raise ValueError(f"TT checkpoint byte size mismatch: {weights}")
    if (
        verify_hash
        and manifest.cores_sha256
        and file_sha256(weights) != manifest.cores_sha256
    ):
        raise ValueError(f"TT checkpoint SHA-256 mismatch: {weights}")
    tensors = load_file(str(weights), device=str(device))
    expected_keys = {f"core_{i}" for i in range(manifest.spec.order)}
    if set(tensors) != expected_keys:
        raise ValueError(f"unexpected core tensor keys in {weights}: {sorted(tensors)}")
    cores = [tensors[f"core_{i}"] for i in range(manifest.spec.order)]
    validate_cores(cores, manifest.spec)
    return cores, manifest


def save_module_set_index(
    root: str | Path,
    entries: Sequence[TTModuleSetEntry],
    *,
    model_path: str = "",
    purpose: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> TTModuleSetManifest:
    manifest = TTModuleSetManifest(
        tuple(sorted(entries, key=lambda e: e.module_path)),
        model_path,
        purpose,
        dict(metadata or {}),
    )
    atomic_write_json(Path(root) / "index.json", manifest.to_dict())
    return manifest


def load_module_set_index(
    root_or_index: str | Path, *, verify_modules: bool = True
) -> tuple[Path, TTModuleSetManifest]:
    value = Path(root_or_index)
    index_path = value if value.name == "index.json" else value / "index.json"
    root = index_path.parent
    if not index_path.is_file():
        raise FileNotFoundError(f"TT module-set index does not exist: {index_path}")
    manifest = TTModuleSetManifest.from_dict(load_json(index_path))
    if verify_modules:
        for entry in manifest.modules:
            directory = root / entry.checkpoint
            module_manifest = TTModuleManifest.from_dict(
                load_json(directory / "manifest.json")
            )
            if module_manifest.module_path != entry.module_path:
                raise ValueError(f"module path mismatch for {directory}")
            weights = directory / "cores.safetensors"
            if (
                not weights.is_file()
                or weights.stat().st_size != module_manifest.checkpoint_bytes
            ):
                raise ValueError(f"invalid core file for {entry.module_path}")
            actual = file_sha256(weights)
            if actual != entry.cores_sha256:
                raise ValueError(
                    f"module-set core SHA-256 mismatch for {entry.module_path}"
                )
    return root, manifest


def publish_module_set(temporary: Path, target: Path) -> None:
    backup: Path | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists():
            backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex}"
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
