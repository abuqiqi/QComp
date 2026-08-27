"""Backend-isolated TT layer and full-training benchmarks."""

from __future__ import annotations
import gc, statistics, time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from .backends import available_tt_backends, create_tt_linear
from .checkpoint import load_module_set_index, load_tt_module
from .model import install_tt_modules
from .provenance import atomic_write_json
from .training import freeze_except_tt


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    return {
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
    }


def _stats(samples: Sequence[float]) -> dict[str, Any]:
    return {
        "samples_ms": list(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


@dataclass(frozen=True)
class LayerBenchmarkConfig:
    module_set: Path
    module_path: str
    backends: tuple[str, ...] = ("native", "tensorly_torch")
    device: str = "cuda:0"
    tokens: int = 128
    token_chunk_size: int = 8
    warmup: int = 3
    iterations: int = 10
    training_dtype: torch.dtype = torch.float32
    activation_checkpointing: bool = False


def benchmark_tt_layer(
    config: LayerBenchmarkConfig, *, output_path: str | Path | None = None
) -> dict[str, Any]:
    if (
        config.tokens <= 0
        or config.iterations <= 0
        or config.token_chunk_size <= 0
        or config.warmup < 0
    ):
        raise ValueError("invalid layer benchmark dimensions")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but unavailable")
    root, index = load_module_set_index(config.module_set, verify_modules=False)
    entries = {entry.module_path: entry for entry in index.modules}
    if config.module_path not in entries:
        raise KeyError(f"module path is absent from module set: {config.module_path}")
    initial_cores, manifest = load_tt_module(
        root / entries[config.module_path].checkpoint
    )
    initial_cores = [
        core.to(device=device, dtype=config.training_dtype) for core in initial_cores
    ]
    input_reference = torch.randn(
        config.tokens,
        manifest.spec.in_features,
        generator=torch.Generator(device="cpu").manual_seed(42),
        dtype=config.training_dtype,
    ).to(device)
    results: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    for backend in config.backends:
        if backend not in available_tt_backends():
            raise ValueError(f"unknown backend: {backend}")
        try:
            layer = create_tt_linear(
                manifest.spec,
                initial_cores,
                tt_backend=backend,
                token_chunk_size=config.token_chunk_size,
                trainable=True,
                activation_checkpointing=config.activation_checkpointing,
            ).to(device)
            layer.train()
            inputs = input_reference.detach().clone().requires_grad_(True)

            def forward_backward() -> Tensor:
                layer.zero_grad(set_to_none=True)
                inputs.grad = None
                value = layer(inputs)
                value.square().mean().backward()
                return value

            for _ in range(config.warmup):
                forward_backward()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            forward_samples: list[float] = []
            backward_samples: list[float] = []
            with torch.no_grad():
                for _ in range(config.iterations):
                    _sync(device)
                    started = time.perf_counter()
                    output = layer(inputs)
                    _sync(device)
                    forward_samples.append((time.perf_counter() - started) * 1000)
            for _ in range(config.iterations):
                _sync(device)
                started = time.perf_counter()
                output = forward_backward()
                _sync(device)
                backward_samples.append((time.perf_counter() - started) * 1000)
            current = {
                "output": output.detach().cpu(),
                "input_gradient": inputs.grad.detach().cpu(),
                "core_gradients": [
                    p.grad.detach().cpu() for p in layer.tt_parameters()
                ],
            }
            if reference is None:
                reference = current
                errors = {
                    "output_max_abs": 0.0,
                    "input_gradient_max_abs": 0.0,
                    "core_gradient_max_abs": [0.0] * len(current["core_gradients"]),
                }
            else:
                errors = {
                    "output_max_abs": float(
                        (current["output"] - reference["output"]).abs().max()
                    ),
                    "input_gradient_max_abs": float(
                        (current["input_gradient"] - reference["input_gradient"])
                        .abs()
                        .max()
                    ),
                    "core_gradient_max_abs": [
                        float((a - b).abs().max())
                        for a, b in zip(
                            current["core_gradients"],
                            reference["core_gradients"],
                            strict=True,
                        )
                    ],
                }
            results[backend] = {
                "status": "ok",
                "backend_metadata": layer.backend_metadata(),
                "forward": {
                    **_stats(forward_samples),
                    "tokens_per_second": config.tokens
                    / (statistics.median(forward_samples) / 1000),
                },
                "forward_backward": {
                    **_stats(backward_samples),
                    "tokens_per_second": config.tokens
                    / (statistics.median(backward_samples) / 1000),
                },
                "errors_vs_first_backend": errors,
                **_memory(device),
            }
            del layer, inputs, output
        except Exception as error:
            results[backend] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            _sync(device)
    payload = {
        "format": "qwen3-tn-layer-benchmark-v1",
        "config": {
            **asdict(config),
            "module_set": str(config.module_set),
            "training_dtype": str(config.training_dtype),
        },
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    module_set: Path
    backends: tuple[str, ...] = ("native", "tensorly_torch")
    device: str = "cuda:0"
    steps: int = 5
    warmup_steps: int = 1
    gradient_accumulation_steps: int = 8
    token_chunk_size: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    bf16_autocast: bool = True
    gradient_checkpointing: bool = True
    tt_activation_checkpointing: bool = True


def benchmark_tt_training(
    config: TrainingBenchmarkConfig,
    model_factory: Callable[[], nn.Module],
    batch_factory: Callable[[nn.Module, torch.device], Mapping[str, Tensor]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if (
        config.steps <= 0
        or config.warmup_steps < 0
        or config.gradient_accumulation_steps <= 0
    ):
        raise ValueError("invalid training benchmark step counts")
    device = torch.device(config.device)
    results: dict[str, Any] = {}
    for backend in config.backends:
        try:
            model = model_factory().to(device)
            modules = install_tt_modules(
                model,
                config.module_set,
                tt_backend=backend,
                trainable=True,
                core_dtype=torch.float32,
                token_chunk_size=config.token_chunk_size,
            )
            summary = freeze_except_tt(model)
            for module in modules.values():
                module.set_activation_checkpointing(config.tt_activation_checkpointing)
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if config.gradient_checkpointing:
                enable = getattr(model, "gradient_checkpointing_enable", None)
                if enable is None:
                    raise TypeError("model does not support gradient checkpointing")
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(parameters, lr=config.learning_rate)
            batch = dict(batch_factory(model, device))
            model.train()
            optimizer.zero_grad(set_to_none=True)

            def update() -> float:
                values: list[float] = []
                for _ in range(config.gradient_accumulation_steps):
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and config.bf16_autocast
                        else nullcontext()
                    )
                    with context:
                        output = model(**batch)
                        loss = (
                            output.get("loss")
                            if isinstance(output, Mapping)
                            else output.loss
                        )
                    values.append(float(loss.detach().cpu()))
                    (loss / config.gradient_accumulation_steps).backward()
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return statistics.fmean(values)

            for _ in range(config.warmup_steps):
                update()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples: list[float] = []
            losses: list[float] = []
            for _ in range(config.steps):
                _sync(device)
                started = time.perf_counter()
                losses.append(update())
                _sync(device)
                samples.append((time.perf_counter() - started) * 1000)
            tokens = (
                int(batch["input_ids"].numel()) * config.gradient_accumulation_steps
            )
            results[backend] = {
                "status": "ok",
                "module_paths": sorted(modules),
                "trainable_tt_parameters": summary.trainable_parameters,
                "optimizer_step": {
                    **_stats(samples),
                    "tokens_per_update": tokens,
                    "tokens_per_second": tokens / (statistics.median(samples) / 1000),
                    "mean_loss": statistics.fmean(losses),
                    **_memory(device),
                },
            }
            del model, optimizer, batch, modules
        except Exception as error:
            results[backend] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            _sync(device)
    payload = {
        "format": "qwen3-tn-training-benchmark-v1",
        "config": {**asdict(config), "module_set": str(config.module_set)},
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload
