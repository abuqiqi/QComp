"""Deterministic signatures and small atomic metadata writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model_path: str | Path) -> dict[str, Any]:
    """Sign stable model metadata without hashing multi-gigabyte weights."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    names = ("config.json", "generation_config.json", "tokenizer_config.json")
    files = {
        name: {
            "sha256": file_sha256(root / name),
            "bytes": (root / name).stat().st_size,
        }
        for name in names
        if (root / name).is_file()
    }
    weights = sorted(
        path.name
        for pattern in ("*.safetensors", "*.bin")
        for path in root.glob(pattern)
        if path.is_file()
    )
    descriptor = {"path": str(root), "metadata_files": files, "weight_files": weights}
    return {**descriptor, "sha256": canonical_json_sha256(descriptor)}


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_publish_directory(temporary: Path, target: Path) -> None:
    """Replace a directory atomically and restore the previous value on failure."""

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


def require_signature(actual: Any, expected_sha256: str, *, label: str) -> None:
    actual_sha256 = canonical_json_sha256(actual)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} signature mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
