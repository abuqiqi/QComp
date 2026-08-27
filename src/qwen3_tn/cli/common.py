"""Shared strict configuration parsing for thin command-line adapters."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Mapping
import torch
from ..provenance import load_json

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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


def parse_dtype(value: str) -> torch.dtype:
    try:
        return DTYPES[value]
    except KeyError as error:
        raise ValueError(
            f"unsupported dtype {value!r}; choose from {sorted(DTYPES)}"
        ) from error


def check_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device
