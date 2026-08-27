"""Atomic optimizer, scheduler, RNG, and loop-state checkpoints."""

from __future__ import annotations
import os
import random
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ..model import export_tt_modules, find_tt_modules, load_tt_cores
from ..provenance import atomic_write_json, load_json
from .config import TTFineTuneConfig, TTTrainingState


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _publish_directory(temporary: Path, target: Path) -> None:
    backup: Path | None = None
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


def save_training_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    state: TTTrainingState,
    *,
    model_path: str,
    config: TTFineTuneConfig,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        export_tt_modules(
            model,
            temporary / "tt_modules",
            model_path=model_path,
            purpose="tt-finetune-resume",
            core_dtype=torch.float32,
            metadata={"global_step": state.global_step},
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(),
            },
            temporary / "training_state.pt",
        )
        backend_names = sorted(
            {module.backend_name for module in find_tt_modules(model).values()}
        )
        atomic_write_json(
            temporary / "trainer_state.json",
            {
                "format": "qwen3-tn-training-state-v1",
                "state": asdict(state),
                "config": asdict(config),
                "backend_names": backend_names,
                "metadata": dict(metadata or {}),
            },
        )
        _publish_directory(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _optimizer_state_to_device(optimizer: Optimizer) -> None:
    for parameter, values in optimizer.state.items():
        for key, value in values.items():
            if isinstance(value, Tensor):
                values[key] = value.to(parameter.device)


def load_training_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    *,
    expected_config: TTFineTuneConfig | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> TTTrainingState:
    source = Path(directory)
    descriptor = load_json(source / "trainer_state.json")
    if descriptor.get("format") != "qwen3-tn-training-state-v1":
        raise ValueError("unsupported training checkpoint format")
    if expected_config is not None and descriptor.get("config") != asdict(
        expected_config
    ):
        raise ValueError("training checkpoint config does not match")
    if expected_metadata is not None and descriptor.get("metadata") != dict(
        expected_metadata
    ):
        raise ValueError("training checkpoint metadata does not match")
    installed_backends = sorted(
        {module.backend_name for module in find_tt_modules(model).values()}
    )
    if descriptor.get("backend_names") != installed_backends:
        raise ValueError("optimizer state cannot be restored across TT backends")
    load_tt_cores(
        model, source / "tt_modules" / "index.json", require_same_backend=True
    )
    payload = torch.load(
        source / "training_state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_state_to_device(optimizer)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return TTTrainingState(**descriptor["state"])


def rotate_training_checkpoints(output_dir: str | Path, limit: int) -> None:
    root = Path(output_dir)
    checkpoints = sorted(
        (
            (int(path.name.removeprefix("checkpoint-")), path)
            for path in root.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda item: item[0],
    )
    for _, path in checkpoints[:-limit]:
        shutil.rmtree(path)
