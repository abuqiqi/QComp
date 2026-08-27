"""Generic baseline → TT-before → fine-tune → TT-after workflow."""

from __future__ import annotations
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import torch
from torch import nn
from ..data import DocumentSource, PreparedCausalLMData, prepare_causal_lm_data
from ..evaluation import EvaluationConfig, evaluate_causal_lm, extract_metrics
from ..model import TTModulePatch, load_tt_cores
from ..provenance import (
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    load_json,
    model_signature,
)
from ..training import TTFineTuneConfig, finetune_causal_lm
from .evaluate import (
    EvaluationStageResult,
    evaluate_with_cache,
    evaluation_cache_metadata,
)


@dataclass(frozen=True)
class FineTuneExperimentConfig:
    model_path: Path
    module_set: Path
    artifact_root: Path
    result_root: Path
    training: TTFineTuneConfig
    evaluation: EvaluationConfig | None = None
    tt_backend: str = "native"
    force_retrain: bool = False
    force_reevaluate: bool = False


def _load_default_model(config: FineTuneExperimentConfig) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(config.training.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    return model, tokenizer


def _training_signature(
    config: FineTuneExperimentConfig,
    prepared: PreparedCausalLMData,
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    index = (
        config.module_set
        if config.module_set.name == "index.json"
        else config.module_set / "index.json"
    )
    return {
        "format": "qwen3-tn-finetune-signature-v1",
        "model_signature": dict(signature),
        "module_set_sha256": file_sha256(index),
        "backend": config.tt_backend,
        "data_fingerprint": prepared.fingerprint,
        "data_metadata": dict(prepared.metadata),
        "training": asdict(config.training),
    }


def _completed_training(
    config: FineTuneExperimentConfig, signature: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    path = config.artifact_root / "run.json"
    final_index = config.artifact_root / "final" / "index.json"
    if config.force_retrain or not path.is_file() or not final_index.is_file():
        return None
    try:
        value = load_json(path)
        if value.get("training_signature") != dict(signature) or value.get(
            "final_index_sha256"
        ) != file_sha256(final_index):
            return None
        return value
    except (OSError, ValueError):
        return None


def _comparison(
    baseline: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    evaluation: EvaluationConfig,
) -> dict[str, Any]:
    values = {
        "baseline": extract_metrics(baseline, evaluation),
        "before": extract_metrics(before, evaluation),
        "after": extract_metrics(after, evaluation),
    }
    changes: dict[str, Any] = {}
    for metric in evaluation.metrics:
        dense, old, new = (
            values["baseline"][metric.name],
            values["before"][metric.name],
            values["after"][metric.name],
        )
        degradation_before = old - dense if metric.direction == "lower" else dense - old
        degradation_after = new - dense if metric.direction == "lower" else dense - new
        changes[metric.name] = {
            "direction": metric.direction,
            "after_minus_before": new - old,
            "after_vs_before_percent": 100 * (new - old) / old if old else None,
            "before_degradation": degradation_before,
            "after_degradation": degradation_after,
            "recovered_fraction": (
                (degradation_before - degradation_after) / degradation_before
                if degradation_before
                else None
            ),
        }
    return {
        "format": "qwen3-tn-finetune-comparison-v1",
        "metrics": values,
        "changes": changes,
    }


def run_finetune_experiment(
    config: FineTuneExperimentConfig,
    data: DocumentSource | PreparedCausalLMData,
    *,
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_causal_lm,
) -> dict[str, Any]:
    owns_model = model is None
    if model is None or tokenizer is None:
        if model is not None or tokenizer is not None:
            raise ValueError("model and tokenizer must be supplied together")
        model, tokenizer = _load_default_model(config)
    signature = model_signature(config.model_path)
    prepared = (
        data
        if isinstance(data, PreparedCausalLMData)
        else prepare_causal_lm_data(
            data,
            tokenizer,
            max_length=config.training.max_length,
            batch_size=config.training.per_device_train_batch_size,
            seed=config.training.seed,
            pin_memory=torch.device(config.training.device).type == "cuda",
        )
    )
    baseline: EvaluationStageResult | None = None
    before: EvaluationStageResult | None = None
    after: EvaluationStageResult | None = None
    try:
        if config.evaluation is not None:
            metadata = evaluation_cache_metadata(
                variant="dense", model_signature=signature, config=config.evaluation
            )
            baseline = evaluate_with_cache(
                model,
                tokenizer,
                config.evaluation,
                config.result_root / "baseline.json",
                metadata,
                evaluator=evaluator,
                force=config.force_reevaluate,
            )
        with TTModulePatch(
            model,
            config.module_set,
            tt_backend=config.tt_backend,
            trainable=True,
            core_dtype=torch.float32,
        ) as modules:
            if config.evaluation is not None:
                metadata = evaluation_cache_metadata(
                    variant="tt-before",
                    model_signature=signature,
                    config=config.evaluation,
                    module_set=config.module_set,
                    backend=config.tt_backend,
                )
                before = evaluate_with_cache(
                    model,
                    tokenizer,
                    config.evaluation,
                    config.result_root / "before.json",
                    metadata,
                    evaluator=evaluator,
                    force=config.force_reevaluate,
                )
            training_signature = _training_signature(config, prepared, signature)
            completed = _completed_training(config, training_signature)
            if completed is not None:
                load_tt_cores(model, config.artifact_root / "final" / "index.json")
                training_metrics = dict(completed["training_metrics"])
                training_source = "checkpoint"
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                training_metrics = finetune_causal_lm(
                    model,
                    prepared.dataloader,
                    config.training,
                    config.artifact_root,
                    model_path=str(config.model_path.resolve()),
                    checkpoint_metadata={
                        "training_signature_sha256": canonical_json_sha256(
                            training_signature
                        )
                    },
                )
                training_source = "training"
                run = {
                    "format": "qwen3-tn-finetune-run-v1",
                    "training_signature": training_signature,
                    "final_index_sha256": file_sha256(
                        config.artifact_root / "final" / "index.json"
                    ),
                    "training_metrics": training_metrics,
                }
                atomic_write_json(config.artifact_root / "run.json", run)
                load_tt_cores(model, config.artifact_root / "final" / "index.json")
            model.eval()
            if config.evaluation is not None:
                metadata = evaluation_cache_metadata(
                    variant="tt-after",
                    model_signature=signature,
                    config=config.evaluation,
                    module_set=config.artifact_root / "final",
                    backend=config.tt_backend,
                    extra={
                        "training_signature_sha256": canonical_json_sha256(
                            training_signature
                        )
                    },
                )
                after = evaluate_with_cache(
                    model,
                    tokenizer,
                    config.evaluation,
                    config.result_root / "after.json",
                    metadata,
                    evaluator=evaluator,
                    force=config.force_reevaluate,
                )
        result: dict[str, Any] = {
            "format": "qwen3-tn-finetune-experiment-v1",
            "training_source": training_source,
            "training_metrics": training_metrics,
            "data_fingerprint": prepared.fingerprint,
        }
        if config.evaluation is not None and baseline and before and after:
            comparison = _comparison(
                baseline.result, before.result, after.result, config.evaluation
            )
            comparison["sources"] = {
                "baseline": baseline.source,
                "before": before.source,
                "after": after.source,
                "training": training_source,
            }
            atomic_write_json(config.result_root / "comparison.json", comparison)
            result["comparison"] = comparison
        return result
    finally:
        if owns_model:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
