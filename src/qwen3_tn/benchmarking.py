"""Backend-isolated TT layer and full-training benchmarks."""

from __future__ import annotations
import gc, statistics, time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from .backends import create_tt_linear, registered_tt_backends
from .checkpoint import load_module_set_index, load_tt_module
from .model import install_tt_modules
from .provenance import atomic_write_json
from .training import freeze_except_tt


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)


def _error_result(error: Exception) -> dict[str, str]:
    return {
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
    }


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


def _options_for(
    values: Mapping[str, Mapping[str, Any]], backend: str
) -> dict[str, Any]:
    options = values.get(backend, {})
    if not isinstance(options, Mapping):
        raise ValueError(f"backend options for {backend!r} must be an object")
    return dict(options)


def _relative_l2(actual: Tensor, expected: Tensor) -> float:
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float())
    return float(difference / denominator) if denominator else float(difference)


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
    relative_l2_threshold: float = 5e-3
    backend_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


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
        if backend not in registered_tt_backends():
            raise ValueError(f"unknown backend: {backend}")
        try:
            options = _options_for(config.backend_options, backend)
            inference_layer = create_tt_linear(
                manifest.spec,
                initial_cores,
                tt_backend=backend,
                token_chunk_size=config.token_chunk_size,
                trainable=False,
                backend_options=options,
            ).to(device)
            inference_layer.eval()
            inference_inputs = input_reference.detach().clone()
            with torch.inference_mode():
                for _ in range(config.warmup):
                    inference_layer(inference_inputs)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                forward_samples: list[float] = []
                for _ in range(config.iterations):
                    _sync(device)
                    started = time.perf_counter()
                    inference_output = inference_layer(inference_inputs)
                    _sync(device)
                    forward_samples.append((time.perf_counter() - started) * 1000)
            inference_memory = _memory(device)
            inference_output_cpu = inference_output.detach().cpu()
            metadata = inference_layer.backend_metadata()
            del inference_layer, inference_inputs, inference_output
            _release_memory(device)

            training_layer = create_tt_linear(
                manifest.spec,
                initial_cores,
                tt_backend=backend,
                token_chunk_size=config.token_chunk_size,
                trainable=True,
                activation_checkpointing=config.activation_checkpointing,
                backend_options=options,
            ).to(device)
            training_layer.train()
            inputs = input_reference.detach().clone().requires_grad_(True)

            def forward_backward() -> Tensor:
                training_layer.zero_grad(set_to_none=True)
                inputs.grad = None
                value = training_layer(inputs)
                value.square().mean().backward()
                return value

            for _ in range(config.warmup):
                forward_backward()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            backward_samples: list[float] = []
            for _ in range(config.iterations):
                _sync(device)
                started = time.perf_counter()
                training_output = forward_backward()
                _sync(device)
                backward_samples.append((time.perf_counter() - started) * 1000)
            training_memory = _memory(device)
            current = {
                "output": inference_output_cpu,
                "input_gradient": inputs.grad.detach().cpu(),
                "core_gradients": [
                    p.grad.detach().cpu() for p in training_layer.tt_parameters()
                ],
            }
            if reference is None:
                reference = current
                errors = {
                    "output_max_abs": 0.0,
                    "output_relative_l2": 0.0,
                    "input_gradient_max_abs": 0.0,
                    "core_gradient_max_abs": [0.0] * len(current["core_gradients"]),
                    "numerical_pass": True,
                }
            else:
                errors = {
                    "output_max_abs": float(
                        (current["output"] - reference["output"]).abs().max()
                    ),
                    "output_relative_l2": _relative_l2(
                        current["output"], reference["output"]
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
                    "numerical_pass": False,
                }
                errors["numerical_pass"] = (
                    errors["output_relative_l2"] <= config.relative_l2_threshold
                )
            results[backend] = {
                "status": "ok",
                "backend_metadata": {**metadata, "options": options},
                "forward": {
                    **_stats(forward_samples),
                    "tokens_per_second": config.tokens
                    / (statistics.median(forward_samples) / 1000),
                    **inference_memory,
                },
                "forward_backward": {
                    **_stats(backward_samples),
                    "tokens_per_second": config.tokens
                    / (statistics.median(backward_samples) / 1000),
                    **training_memory,
                },
                "errors_vs_first_backend": errors,
            }
            del training_layer, inputs, training_output
        except Exception as error:
            results[backend] = _error_result(error)
        _release_memory(device)
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
    backend_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


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
            options = _options_for(config.backend_options, backend)
            model = model_factory().to(device)
            modules = install_tt_modules(
                model,
                config.module_set,
                tt_backend=backend,
                trainable=True,
                core_dtype=torch.float32,
                token_chunk_size=config.token_chunk_size,
                backend_options=options,
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
                "backend_options": options,
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
            results[backend] = _error_result(error)
        _release_memory(device)
    payload = {
        "format": "qwen3-tn-training-benchmark-v1",
        "config": {**asdict(config), "module_set": str(config.module_set)},
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload


@dataclass(frozen=True)
class InferenceBenchmarkConfig:
    module_set: Path
    backends: tuple[str, ...] = ("native", "tensorly_torch")
    device: str = "cuda:0"
    batch_size: int = 1
    prefill_tokens: int = 1024
    decode_steps: int = 32
    warmup: int = 1
    iterations: int = 3
    core_dtype: torch.dtype = torch.bfloat16
    seed: int = 42
    required_speedup: float = 2.0
    relative_l2_threshold: float = 5e-3
    top1_agreement_threshold: float = 0.99
    backend_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "native" not in self.backends:
            raise ValueError("inference benchmark requires native as the reference")
        if (
            self.batch_size <= 0
            or self.prefill_tokens <= 0
            or self.decode_steps <= 0
            or self.iterations <= 0
            or self.warmup < 0
        ):
            raise ValueError("invalid inference benchmark dimensions")
        if self.required_speedup <= 0:
            raise ValueError("required_speedup must be positive")
        if self.relative_l2_threshold < 0:
            raise ValueError("relative_l2_threshold must be non-negative")
        if not 0 <= self.top1_agreement_threshold <= 1:
            raise ValueError("top1_agreement_threshold must be in [0, 1]")


def _model_values(output: Any) -> tuple[Tensor, Any]:
    logits = output.get("logits") if isinstance(output, Mapping) else output.logits
    cache = (
        output.get("past_key_values")
        if isinstance(output, Mapping)
        else output.past_key_values
    )
    if not isinstance(logits, Tensor) or cache is None:
        raise TypeError("model output must contain logits and past_key_values")
    return logits, cache


def _prefill(model: nn.Module, input_ids: Tensor) -> tuple[Tensor, Any]:
    return _model_values(model(input_ids=input_ids, use_cache=True))


def _decode(
    model: nn.Module, cache: Any, decode_ids: Tensor
) -> tuple[Tensor, Tensor]:
    logits: list[Tensor] = []
    top1: list[Tensor] = []
    for step in range(decode_ids.shape[1]):
        output = model(
            input_ids=decode_ids[:, step : step + 1],
            past_key_values=cache,
            use_cache=True,
        )
        value, cache = _model_values(output)
        last = value[:, -1, :]
        logits.append(last)
        top1.append(last.argmax(dim=-1))
    return torch.stack(logits, dim=1), torch.stack(top1, dim=1)


def _throughput(tokens: int, samples_ms: Sequence[float]) -> float:
    return tokens / (statistics.median(samples_ms) / 1000)


def benchmark_tt_inference(
    config: InferenceBenchmarkConfig,
    model_factory: Callable[[], nn.Module],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark full-model TT prefill and fixed-token KV-cache decode."""

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but unavailable")
    prompt_reference: Tensor | None = None
    decode_reference: Tensor | None = None
    results: dict[str, Any] = {}
    traces: dict[str, tuple[Tensor, Tensor]] = {}
    for backend in config.backends:
        try:
            options = _options_for(config.backend_options, backend)
            model = model_factory().to(device)
            modules = install_tt_modules(
                model,
                config.module_set,
                tt_backend=backend,
                trainable=False,
                core_dtype=config.core_dtype,
                backend_options=options,
            )
            model.eval()
            vocab_size = int(model.config.vocab_size)
            if prompt_reference is None:
                generator = torch.Generator().manual_seed(config.seed)
                prompt_reference = torch.randint(
                    0,
                    vocab_size,
                    (config.batch_size, config.prefill_tokens),
                    generator=generator,
                )
                decode_reference = torch.randint(
                    0,
                    vocab_size,
                    (config.batch_size, config.decode_steps),
                    generator=generator,
                )
            prompt = prompt_reference.to(device)
            decode_ids = decode_reference.to(device)
            _release_memory(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                for _ in range(config.warmup):
                    _, cache = _prefill(model, prompt)
                    _decode(model, cache, decode_ids)
                prefill_samples: list[float] = []
                for _ in range(config.iterations):
                    _sync(device)
                    started = time.perf_counter()
                    _prefill(model, prompt)
                    _sync(device)
                    prefill_samples.append((time.perf_counter() - started) * 1000)
                decode_samples: list[float] = []
                for _ in range(config.iterations):
                    _, cache = _prefill(model, prompt)
                    _sync(device)
                    started = time.perf_counter()
                    _decode(model, cache, decode_ids)
                    _sync(device)
                    decode_samples.append((time.perf_counter() - started) * 1000)
                end_to_end_samples: list[float] = []
                trace_logits: Tensor | None = None
                trace_top1: Tensor | None = None
                for _ in range(config.iterations):
                    _sync(device)
                    started = time.perf_counter()
                    prefill_logits, cache = _prefill(model, prompt)
                    decode_logits, decode_top1 = _decode(model, cache, decode_ids)
                    _sync(device)
                    end_to_end_samples.append(
                        (time.perf_counter() - started) * 1000
                    )
                    last_prefill = prefill_logits[:, -1, :]
                    trace_logits = torch.cat(
                        (last_prefill.unsqueeze(1), decode_logits), dim=1
                    )
                    trace_top1 = torch.cat(
                        (last_prefill.argmax(dim=-1).unsqueeze(1), decode_top1),
                        dim=1,
                    )
            assert trace_logits is not None and trace_top1 is not None
            traces[backend] = (
                trace_logits.float().cpu(),
                trace_top1.cpu(),
            )
            model_memory = _memory(device)
            results[backend] = {
                "status": "ok",
                "backend_metadata": {
                    **next(iter(modules.values())).backend_metadata(),
                    "options": options,
                },
                "module_count": len(modules),
                "prefill": {
                    **_stats(prefill_samples),
                    "tokens_per_second": _throughput(
                        config.batch_size * config.prefill_tokens, prefill_samples
                    ),
                },
                "decode": {
                    **_stats(decode_samples),
                    "tokens_per_second": _throughput(
                        config.batch_size * config.decode_steps, decode_samples
                    ),
                },
                "end_to_end": {
                    **_stats(end_to_end_samples),
                    "tokens_per_second": _throughput(
                        config.batch_size
                        * (config.prefill_tokens + config.decode_steps),
                        end_to_end_samples,
                    ),
                },
                **model_memory,
            }
            del model, modules, prompt, decode_ids, trace_logits, trace_top1
        except Exception as error:
            results[backend] = _error_result(error)
        _release_memory(device)
    reference_result = results.get("native", {})
    reference_trace = traces.get("native")
    if reference_result.get("status") == "ok" and reference_trace is not None:
        native_ms = float(reference_result["end_to_end"]["median_ms"])
        for backend, result in results.items():
            if result.get("status") != "ok":
                continue
            logits, top1 = traces[backend]
            relative_l2 = _relative_l2(logits, reference_trace[0])
            agreement = float((top1 == reference_trace[1]).float().mean())
            speedup = native_ms / float(result["end_to_end"]["median_ms"])
            numerical_pass = (
                relative_l2 <= config.relative_l2_threshold
                and agreement >= config.top1_agreement_threshold
            )
            result["comparison_vs_native"] = {
                "output_relative_l2": relative_l2,
                "top1_agreement": agreement,
                "speedup": speedup,
                "numerical_pass": numerical_pass,
                "performance_pass": speedup >= config.required_speedup,
                "eligible_for_lm_eval": (
                    backend != "native"
                    and numerical_pass
                    and speedup >= config.required_speedup
                ),
            }
    payload = {
        "format": "qwen3-tn-inference-benchmark-v1",
        "config": {
            **asdict(config),
            "module_set": str(config.module_set),
            "core_dtype": str(config.core_dtype),
        },
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload
