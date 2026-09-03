"""Shared strict configuration parsing for thin command-line adapters."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Mapping
import torch
from ..backends import import_tt_backend_modules
from ..model_loading import parse_torch_dtype as parse_dtype
from ..provenance import load_json


def config_argument(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def strict_config(
    path: str | Path, *, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    value = load_json(path)
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    return value


def check_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_backend_modules(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("backend_modules must be a list of module names")
    return import_tt_backend_modules(value)


def parse_backend_options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("backend_options must be an object")
    return dict(value)
