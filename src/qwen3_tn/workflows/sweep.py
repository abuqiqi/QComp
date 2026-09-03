"""Generic full-rank-tuple sweep over arbitrary module paths."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import torch
from torch import nn
from ..model import TTModulePatch, TTTarget
from ..provenance import atomic_write_json
from ..tt import TTMatrixSpec
from .decompose import DecomposeConfig, decompose_targets


@dataclass(frozen=True)
class RankSweepCandidate:
    name: str
    ranks: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if not self.ranks:
            raise ValueError("candidate ranks must not be empty")


@dataclass(frozen=True)
class RankSweepConfig:
    model_path: str
    output_root: Path
    cache_root: Path
    result_path: Path
    candidates: tuple[RankSweepCandidate, ...]
    backend: str = "native"
    core_dtype: torch.dtype = torch.bfloat16
    svd_driver: str | None = "gesvdj"
    selection_metric: str | None = None
    selection_direction: str = "lower"
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("rank sweep candidates must not be empty")
        if len({candidate.name for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate names must be unique")
        if self.selection_direction not in ("lower", "higher"):
            raise ValueError("selection_direction must be lower or higher")


def run_rank_sweep(
    model: nn.Module,
    targets: Sequence[TTTarget],
    config: RankSweepConfig,
    evaluator: Callable[[nn.Module], Mapping[str, Any]],
) -> dict[str, Any]:
    if not targets:
        raise ValueError("targets must not be empty")
    target_paths = {target.module_path for target in targets}
    baseline = dict(evaluator(model))
    candidates: dict[str, Any] = {}
    for candidate in config.candidates:
        if set(candidate.ranks) != target_paths:
            candidates[candidate.name] = {
                "status": "error",
                "error_type": "ValueError",
                "error": "candidate rank paths do not match target paths",
            }
            continue
        try:
            derived = [
                TTTarget(
                    target.module_path,
                    TTMatrixSpec(
                        target.spec.out_modes,
                        target.spec.in_modes,
                        tuple(candidate.ranks[target.module_path]),
                    ),
                    target.token_chunk_size,
                )
                for target in targets
            ]
            artifact = config.output_root / "candidates" / candidate.name
            decomposition = decompose_targets(
                model,
                derived,
                DecomposeConfig(
                    config.model_path,
                    artifact,
                    config.cache_root,
                    config.svd_driver,
                    torch.float32,
                    config.core_dtype,
                    "rank-sweep-candidate",
                ),
            )
            with TTModulePatch(
                model,
                artifact,
                tt_backend=config.backend,
                core_dtype=config.core_dtype,
                backend_options=config.backend_options,
            ):
                metrics = dict(evaluator(model))
            candidates[candidate.name] = {
                "status": "ok",
                "ranks": {path: list(ranks) for path, ranks in candidate.ranks.items()},
                "decomposition": decomposition,
                "metrics": metrics,
            }
        except Exception as error:
            candidates[candidate.name] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    selected: str | None = None
    if config.selection_metric is not None:
        valid = [
            (name, value["metrics"].get(config.selection_metric))
            for name, value in candidates.items()
            if value.get("status") == "ok"
            and isinstance(
                value.get("metrics", {}).get(config.selection_metric), (int, float)
            )
        ]
        if valid:
            selected = (min if config.selection_direction == "lower" else max)(
                valid, key=lambda item: item[1]
            )[0]
    result = {
        "format": "qwen3-tn-rank-sweep-v1",
        "baseline": baseline,
        "candidates": candidates,
        "selection": {
            "metric": config.selection_metric,
            "direction": config.selection_direction,
            "candidate": selected,
        },
    }
    atomic_write_json(config.result_path, result)
    return result
