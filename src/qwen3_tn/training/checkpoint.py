"""Atomic optimizer, scheduler, RNG, and loop-state checkpoints."""

from __future__ import annotations
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
from ..provenance import atomic_publish_directory, atomic_write_json, load_json
from .config import TTFineTuneConfig, TTTrainingState


def _checkpoint_step(path: Path) -> int | None:
    value = path.name.removeprefix("checkpoint-")
    return (
        int(value) if path.name.startswith("checkpoint-") and value.isdigit() else None
    )


def _config_matches(value: Any, expected: TTFineTuneConfig) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_values = asdict(expected)
    if set(value) - set(expected_values):
        return False
    return all(
        value[key] == expected_value if key in value else expected_value is None
        for key, expected_value in expected_values.items()
    )


def latest_training_checkpoint(
    output_dir: str | Path,
    *,
    expected_config: TTFineTuneConfig | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the highest-step complete checkpoint matching this experiment."""

    candidates: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is None or not path.is_dir():
            continue
        try:
            descriptor = load_json(path / "trainer_state.json")
        except (FileNotFoundError, OSError, ValueError):
            continue
        if descriptor.get("format") != "qwen3-tn-training-state-v1":
            continue
        if expected_config is not None and not _config_matches(
            descriptor.get("config"), expected_config
        ):
            continue
        if expected_metadata is not None and descriptor.get("metadata") != dict(
            expected_metadata
        ):
            continue
        if (
            not (path / "training_state.pt").is_file()
            or not (path / "tt_modules" / "index.json").is_file()
        ):
            continue
        candidates.append((step, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


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
        atomic_publish_directory(temporary, target)
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
    if expected_config is not None and not _config_matches(
        descriptor.get("config"), expected_config
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
