"""Resumable evaluation of several model variants on a shared task suite."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .backends import import_tt_backend_modules
from .evaluation import EvaluationConfig, evaluate_causal_lm, extract_metrics
from .model_loading import (
    load_local_causal_lm,
    load_local_tokenizer,
    parse_torch_dtype,
)
from .model import TTModulePatch
from .provenance import atomic_write_json, load_json, model_signature
from .workflows.evaluate import evaluate_with_cache, evaluation_cache_metadata


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SuiteTask:
    name: str
    evaluation: Path

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuiteTask":
        unknown = set(value) - {"name", "evaluation"}
        if unknown:
            raise ValueError(f"unknown suite task fields: {sorted(unknown)}")
        return cls(str(value["name"]), Path(value["evaluation"]))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("suite task name must not be empty")


@dataclass(frozen=True)
class SuiteVariant:
    name: str
    module_set: Path | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuiteVariant":
        unknown = set(value) - {"name", "module_set"}
        if unknown:
            raise ValueError(f"unknown suite variant fields: {sorted(unknown)}")
        module_set = value.get("module_set")
        return cls(
            str(value["name"]), Path(module_set) if module_set is not None else None
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("suite variant name must not be empty")


@dataclass(frozen=True)
class EvaluationSuiteConfig:
    model_path: Path
    output_root: Path
    tasks: tuple[SuiteTask, ...]
    variants: tuple[SuiteVariant, ...]
    comparison: Mapping[str, str]
    backend: str = "native"
    device: str = "cuda:0"
    model_dtype: str = "bfloat16"
    force: bool = False
    backend_modules: tuple[str, ...] = ()
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationSuiteConfig":
        allowed = {
            "model_path", "output_root", "tasks", "variants", "comparison",
            "backend", "device", "model_dtype", "force",
            "backend_modules", "backend_options",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown evaluation suite fields: {sorted(unknown)}")
        required = {"model_path", "output_root", "tasks", "variants", "comparison"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing evaluation suite fields: {sorted(missing)}")
        backend_modules = value.get("backend_modules", ())
        if not isinstance(backend_modules, (list, tuple)) or not all(
            isinstance(item, str) for item in backend_modules
        ):
            raise ValueError("backend_modules must be a list of module names")
        backend_options = value.get("backend_options", {})
        if not isinstance(backend_options, Mapping):
            raise ValueError("backend_options must be an object")
        return cls(
            model_path=Path(value["model_path"]),
            output_root=Path(value["output_root"]),
            tasks=tuple(SuiteTask.from_dict(item) for item in value["tasks"]),
            variants=tuple(SuiteVariant.from_dict(item) for item in value["variants"]),
            comparison=dict(value["comparison"]),
            backend=str(value.get("backend", "native")),
            device=str(value.get("device", "cuda:0")),
            model_dtype=str(value.get("model_dtype", "bfloat16")),
            force=bool(value.get("force", False)),
            backend_modules=tuple(backend_modules),
            backend_options=dict(backend_options),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationSuiteConfig":
        return cls.from_dict(load_json(path))

    def __post_init__(self) -> None:
        if not self.tasks or not self.variants:
            raise ValueError("evaluation suite needs at least one task and variant")
        task_names = [task.name for task in self.tasks]
        variant_names = [variant.name for variant in self.variants]
        if len(task_names) != len(set(task_names)):
            raise ValueError("suite task names must be unique")
        if len(variant_names) != len(set(variant_names)):
            raise ValueError("suite variant names must be unique")
        roles = {"baseline", "compressed", "retrained"}
        if set(self.comparison) != roles:
            raise ValueError(f"comparison must define exactly {sorted(roles)}")
        missing = set(self.comparison.values()) - set(variant_names)
        if missing:
            raise ValueError(f"comparison refers to unknown variants: {sorted(missing)}")
        if len(set(self.comparison.values())) != 3:
            raise ValueError("comparison roles must refer to three distinct variants")


def _read_result_metrics(
    path: Path, evaluation: EvaluationConfig
) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        return extract_metrics(load_json(path), evaluation)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def build_suite_summary(
    config: EvaluationSuiteConfig,
    evaluations: Mapping[str, EvaluationConfig],
    *,
    started_at: str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build a compact comparison from any completed per-task result files."""
    scores: dict[str, dict[str, dict[str, float]]] = {}
    completed: list[str] = []
    for variant in config.variants:
        variant_scores: dict[str, dict[str, float]] = {}
        for task in config.tasks:
            metrics = _read_result_metrics(
                config.output_root / variant.name / f"{task.name}.json",
                evaluations[task.name],
            )
            if metrics is not None:
                variant_scores[task.name] = metrics
                completed.append(f"{variant.name}/{task.name}")
        scores[variant.name] = variant_scores

    base_name = config.comparison["baseline"]
    compressed_name = config.comparison["compressed"]
    retrained_name = config.comparison["retrained"]
    comparisons: list[dict[str, Any]] = []
    for task in config.tasks:
        evaluation = evaluations[task.name]
        for metric in evaluation.metrics:
            baseline = scores.get(base_name, {}).get(task.name, {}).get(metric.name)
            compressed = scores.get(compressed_name, {}).get(task.name, {}).get(metric.name)
            retrained = scores.get(retrained_name, {}).get(task.name, {}).get(metric.name)
            row: dict[str, Any] = {
                "task": task.name,
                "lm_eval_task": evaluation.task,
                "metric": metric.name,
                "direction": metric.direction,
                "baseline": baseline,
                "compressed": compressed,
                "retrained": retrained,
                "compressed_delta": None,
                "retrained_delta": None,
                "healing_gain": None,
                "recovery_fraction": None,
            }
            if baseline is not None and compressed is not None:
                row["compressed_delta"] = compressed - baseline
            if baseline is not None and retrained is not None:
                row["retrained_delta"] = retrained - baseline
            if compressed is not None and retrained is not None:
                row["healing_gain"] = retrained - compressed
            if all(value is not None for value in (baseline, compressed, retrained)):
                if metric.direction == "lower":
                    lost = compressed - baseline
                    regained = compressed - retrained
                else:
                    lost = baseline - compressed
                    regained = retrained - compressed
                if lost > 0:
                    row["recovery_fraction"] = regained / lost
            comparisons.append(row)

    expected = len(config.tasks) * len(config.variants)
    return {
        "format": "qwen3-tn-evaluation-suite-summary-v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "complete": len(completed) == expected,
        "completed_evaluations": completed,
        "expected_evaluations": expected,
        "model_path": str(config.model_path.resolve()),
        "backend": config.backend,
        "backend_options": dict(config.backend_options),
        "comparison": dict(config.comparison),
        "scores": scores,
        "comparisons": comparisons,
    }


def render_suite_markdown(summary: Mapping[str, Any]) -> str:
    roles = summary["comparison"]

    def score(value: Any) -> str:
        return "—" if value is None else f"{100 * float(value):.2f}"

    def delta(value: Any) -> str:
        return "—" if value is None else f"{100 * float(value):+.2f}"

    def recovery(value: Any) -> str:
        return "—" if value is None else f"{100 * float(value):.1f}%"

    lines = [
        "# CompactifAI-aligned evaluation", "",
        f"Status: {'complete' if summary['complete'] else 'running/incomplete'} "
        f"({len(summary['completed_evaluations'])}/{summary['expected_evaluations']})",
        "",
        "Scores and deltas are percentage points. Recovery is the fraction of the "
        "baseline-to-compressed loss recovered by retraining.", "",
        f"| Task / metric | {roles['baseline']} | {roles['compressed']} | "
        f"{roles['retrained']} | MPO delta | retrained delta | healing gain | recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['task']} / {row['metric']} | {score(row['baseline'])} | "
            f"{score(row['compressed'])} | {score(row['retrained'])} | "
            f"{delta(row['compressed_delta'])} | {delta(row['retrained_delta'])} | "
            f"{delta(row['healing_gain'])} | {recovery(row['recovery_fraction'])} |"
        )
    lines.extend(["", "Raw lm-eval outputs are stored under each variant directory.", ""])
    return "\n".join(lines)


def _write_progress(
    config: EvaluationSuiteConfig,
    evaluations: Mapping[str, EvaluationConfig],
    *,
    started_at: str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    summary = build_suite_summary(
        config, evaluations, started_at=started_at, completed_at=completed_at
    )
    atomic_write_json(config.output_root / "summary.json", summary)
    target = config.output_root / "comparison.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_suite_markdown(summary), encoding="utf-8")
    return summary


def run_evaluation_suite(
    config: EvaluationSuiteConfig,
    *,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_causal_lm,
) -> dict[str, Any]:
    """Load the dense model once and evaluate all dense/TT variants resumably."""
    import_tt_backend_modules(config.backend_modules)
    config.output_root.mkdir(parents=True, exist_ok=True)
    evaluations = {
        task.name: EvaluationConfig.from_json(task.evaluation) for task in config.tasks
    }
    started_at = _utc_now()
    state_path = config.output_root / "state.json"
    atomic_write_json(
        state_path,
        {"status": "running", "stage": "loading-model", "started_at": started_at},
    )
    _write_progress(config, evaluations, started_at=started_at)
    try:
        dtype = parse_torch_dtype(config.model_dtype)
        signature = model_signature(config.model_path)
        tokenizer = load_local_tokenizer(config.model_path)
        model = load_local_causal_lm(
            config.model_path,
            dtype=dtype,
            device=config.device,
        )
        model.eval()
        for variant in config.variants:
            patch = (
                contextlib.nullcontext()
                if variant.module_set is None
                else TTModulePatch(
                    model,
                    variant.module_set,
                    tt_backend=config.backend,
                    core_dtype=dtype,
                    backend_options=config.backend_options,
                )
            )
            with patch:
                for task in config.tasks:
                    evaluation = evaluations[task.name]
                    stage = f"{variant.name}/{task.name}"
                    atomic_write_json(
                        state_path,
                        {"status": "running", "stage": stage, "started_at": started_at},
                    )
                    print(f"[{_utc_now()}] evaluating {stage}", flush=True)
                    metadata = evaluation_cache_metadata(
                        variant=variant.name,
                        model_signature=signature,
                        config=evaluation,
                        module_set=variant.module_set,
                        backend=config.backend if variant.module_set else None,
                        backend_options=(
                            config.backend_options if variant.module_set else None
                        ),
                        extra={"suite_format": "compactifai-aligned-v1"},
                    )
                    result = evaluate_with_cache(
                        model,
                        tokenizer,
                        evaluation,
                        config.output_root / variant.name / f"{task.name}.json",
                        metadata,
                        evaluator=evaluator,
                        force=config.force,
                    )
                    metrics = extract_metrics(result.result, evaluation)
                    print(
                        f"[{_utc_now()}] completed {stage} "
                        f"source={result.source} metrics={metrics}",
                        flush=True,
                    )
                    _write_progress(config, evaluations, started_at=started_at)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        completed_at = _utc_now()
        summary = _write_progress(
            config, evaluations, started_at=started_at, completed_at=completed_at
        )
        atomic_write_json(
            state_path,
            {
                "status": "success", "stage": "complete",
                "started_at": started_at, "completed_at": completed_at,
                "summary": str((config.output_root / "summary.json").resolve()),
            },
        )
        return summary
    except BaseException as error:
        atomic_write_json(
            state_path,
            {
                "status": "failure", "stage": "evaluation",
                "started_at": started_at, "completed_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
