"""Dense/TT evaluation and exact metadata caches."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import torch
from torch import nn
from ..evaluation import EvaluationConfig, evaluate_causal_lm
from ..model import TTModulePatch
from ..provenance import (
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    load_json,
)


@dataclass(frozen=True)
class EvaluationStageResult:
    result: Mapping[str, Any]
    source: str


def evaluation_cache_metadata(
    *,
    variant: str,
    model_signature: Mapping[str, Any],
    config: EvaluationConfig,
    module_set: str | Path | None = None,
    backend: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": "qwen3-tn-evaluation-cache-v1",
        "variant": variant,
        "model_signature": dict(model_signature),
        "evaluation_config": config.to_dict(),
    }
    if module_set is not None:
        index = Path(module_set)
        index = index if index.name == "index.json" else index / "index.json"
        metadata["tt"] = {
            "backend": backend,
            "module_set_index": str(index.resolve()),
            "index_sha256": file_sha256(index),
        }
    metadata.update(dict(extra or {}))
    metadata["sha256"] = canonical_json_sha256(metadata)
    return metadata


def evaluate_with_cache(
    model: nn.Module,
    tokenizer: Any,
    config: EvaluationConfig,
    cache_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_causal_lm,
    force: bool = False,
) -> EvaluationStageResult:
    target = Path(cache_path)
    if not force and target.is_file():
        try:
            cached = load_json(target)
            if (
                cached.get("cache_metadata") == dict(metadata)
                and "raw_results" in cached
            ):
                return EvaluationStageResult(cached, "cache")
        except (ValueError, OSError):
            pass
    result = dict(evaluator(model, config, tokenizer=tokenizer))
    result["cache_metadata"] = dict(metadata)
    atomic_write_json(target, result)
    return EvaluationStageResult(result, "evaluation")


def evaluate_dense_or_tt(
    model: nn.Module,
    tokenizer: Any,
    config: EvaluationConfig,
    *,
    cache_path: str | Path,
    metadata: Mapping[str, Any],
    module_set: str | Path | None = None,
    backend: str = "native",
    force: bool = False,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_causal_lm,
) -> EvaluationStageResult:
    if module_set is None:
        return evaluate_with_cache(
            model,
            tokenizer,
            config,
            cache_path,
            metadata,
            evaluator=evaluator,
            force=force,
        )
    with TTModulePatch(
        model, module_set, tt_backend=backend, core_dtype=torch.bfloat16
    ):
        return evaluate_with_cache(
            model,
            tokenizer,
            config,
            cache_path,
            metadata,
            evaluator=evaluator,
            force=force,
        )
