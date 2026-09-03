"""One-model causal-LM evaluation with configurable metrics."""

from __future__ import annotations
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from .model_loading import load_local_causal_lm, load_local_tokenizer
from .provenance import atomic_write_json, load_json


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str = "lower"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        if self.direction not in ("lower", "higher"):
            raise ValueError("metric direction must be lower or higher")


@dataclass(frozen=True)
class EvaluationConfig:
    task: str
    metrics: tuple[MetricSpec, ...]
    limit: int | float | None = None
    batch_size: int = 1
    max_length: int = 2048
    num_fewshot: int = 0
    apply_chat_template: bool = False
    bootstrap_iters: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            tuple(
                MetricSpec(**m) if isinstance(m, Mapping) else m for m in self.metrics
            ),
        )
        if not self.task:
            raise ValueError("task must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        if len({metric.name for metric in self.metrics}) != len(self.metrics):
            raise ValueError("metric names must be unique")
        if self.limit is not None:
            valid = (
                isinstance(self.limit, int)
                and not isinstance(self.limit, bool)
                and self.limit > 0
            ) or (isinstance(self.limit, float) and 0 < self.limit < 1)
            if not valid:
                raise ValueError(
                    "limit must be a positive integer, fraction in (0,1), or None"
                )
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        if self.num_fewshot < 0 or self.bootstrap_iters < 0:
            raise ValueError("num_fewshot and bootstrap_iters must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationConfig":
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown evaluation config fields: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationConfig":
        return cls.from_dict(load_json(path))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metrics"] = [asdict(metric) for metric in self.metrics]
        return value


def extract_metrics(
    result: Mapping[str, Any], config: EvaluationConfig
) -> dict[str, float]:
    try:
        task_values = result["raw_results"]["results"][config.task]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"evaluation result does not contain task {config.task!r}"
        ) from error
    extracted: dict[str, float] = {}
    for metric in config.metrics:
        value = task_values.get(metric.name)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"evaluation result does not contain numeric metric {metric.name!r}"
            )
        extracted[metric.name] = float(value)
    return extracted


def evaluate_causal_lm(
    model_or_path: Any,
    config: EvaluationConfig,
    *,
    tokenizer: Any | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        import numpy as np

        np.random.seed(config.seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model_load_seconds: float | None = None
    if isinstance(model_or_path, (str, os.PathLike)):
        model_path = Path(model_or_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {model_path}")
        started = time.perf_counter()
        tokenizer = tokenizer or load_local_tokenizer(model_path)
        model = load_local_causal_lm(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model_load_seconds = time.perf_counter() - started
        model_source = str(model_path)
    else:
        model = model_or_path
        if not callable(getattr(model, "eval", None)):
            raise TypeError("model_or_path must be a model or local path")
        if tokenizer is None:
            raise ValueError("tokenizer is required with a loaded model")
        model_source = type(model).__name__
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    model.eval()
    evaluator_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    started = time.perf_counter()
    raw = lm_eval.simple_evaluate(
        model=evaluator_model,
        tasks=[config.task],
        num_fewshot=config.num_fewshot,
        limit=config.limit,
        apply_chat_template=config.apply_chat_template,
        log_samples=False,
        bootstrap_iters=config.bootstrap_iters,
        random_seed=config.seed,
        numpy_random_seed=config.seed,
        torch_random_seed=config.seed,
        fewshot_random_seed=config.seed,
    )
    if raw is None:
        raise RuntimeError("lm-eval returned no results")
    result = {
        "config": {**config.to_dict(), "model_source": model_source},
        "timing": {
            "model_load_seconds": model_load_seconds,
            "evaluation_seconds": time.perf_counter() - started,
        },
        "raw_results": raw,
    }
    result = json.loads(
        json.dumps(result, ensure_ascii=False, default=handle_non_serializable)
    )
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result
