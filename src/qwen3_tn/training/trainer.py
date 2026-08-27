"""Single-device causal-LM loop that updates only TT cores."""

from __future__ import annotations
import math
import random
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from ..model import export_tt_modules, find_tt_modules
from ..provenance import atomic_write_json
from .checkpoint import (
    load_training_checkpoint,
    rotate_training_checkpoints,
    save_training_checkpoint,
)
from .config import TTFineTuneConfig, TTTrainableSummary, TTTrainingState


def freeze_except_tt(model: nn.Module) -> TTTrainableSummary:
    modules = find_tt_modules(model)
    if not modules:
        raise RuntimeError("model contains no TT modules")
    model.requires_grad_(False)
    core_ids: set[int] = set()
    for module in modules.values():
        for parameter in module.tt_parameters():
            parameter.requires_grad_(True)
            core_ids.add(id(parameter))
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in core_ids
    ]
    if unexpected:
        raise RuntimeError(f"non-TT parameters remain trainable: {unexpected}")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("no TT core parameters were enabled")
    return TTTrainableSummary(
        tuple(modules),
        len(trainable),
        sum(p.numel() for p in trainable),
        sum(p.numel() for p in model.parameters()),
    )


def build_tt_optimizer(model: nn.Module, config: TTFineTuneConfig) -> Optimizer:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    allowed = {
        id(parameter)
        for module in find_tt_modules(model).values()
        for parameter in module.tt_parameters()
    }
    if any(id(parameter) not in allowed for parameter in parameters):
        raise RuntimeError("optimizer would receive non-TT parameters")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise TypeError("all trainable TT cores must be FP32")
    return AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def _scheduler(optimizer: Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return LambdaLR(optimizer, scale)


def _loss(output: Any) -> Tensor:
    value = (
        output.get("loss")
        if isinstance(output, Mapping)
        else getattr(output, "loss", None)
    )
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise TypeError("causal LM output must contain a scalar loss")
    if not torch.isfinite(value):
        raise FloatingPointError(
            f"training loss is not finite: {value.detach().item()}"
        )
    return value


def finetune_causal_lm(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    config: TTFineTuneConfig,
    output_dir: str | Path,
    *,
    model_path: str,
    resume_from: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader must not be empty")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    summary = freeze_except_tt(model)
    modules = find_tt_modules(model)
    for module in modules.values():
        module.set_activation_checkpointing(config.tt_activation_checkpointing)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if config.gradient_checkpointing:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if enable is None:
            raise TypeError("model does not support gradient checkpointing")
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = build_tt_optimizer(model, config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    updates_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    planned_steps = config.max_steps or config.num_train_epochs * updates_per_epoch
    epochs = (
        math.ceil(planned_steps / updates_per_epoch)
        if config.max_steps
        else config.num_train_epochs
    )
    scheduler = _scheduler(optimizer, planned_steps, config.warmup_ratio)
    state = TTTrainingState()
    if resume_from is not None:
        state = load_training_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            expected_config=config,
            expected_metadata=checkpoint_metadata,
        )
    if state.global_step >= planned_steps:
        raise ValueError(
            f"resume step {state.global_step} reached planned steps {planned_steps}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = getattr(train_dataloader, "sampler", None)
    losses: list[float] = []
    stopped = False
    for epoch in range(state.epoch, epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(train_dataloader):
            if epoch == state.epoch and batch_index < state.batch_index:
                continue
            window_start = (
                batch_index
                // config.gradient_accumulation_steps
                * config.gradient_accumulation_steps
            )
            window_size = min(
                config.gradient_accumulation_steps, len(train_dataloader) - window_start
            )
            batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in raw_batch.items()
            }
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and config.bf16_autocast
                else nullcontext()
            )
            with autocast:
                loss = _loss(model(**batch))
            losses.append(float(loss.detach().cpu()))
            (loss / window_size).backward()
            if (
                batch_index + 1
            ) % config.gradient_accumulation_steps and batch_index + 1 != len(
                train_dataloader
            ):
                continue
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            next_epoch, next_batch = epoch, batch_index + 1
            if next_batch == len(train_dataloader):
                next_epoch, next_batch = epoch + 1, 0
            state = TTTrainingState(state.global_step + 1, next_epoch, next_batch)
            if state.global_step % config.logging_steps == 0:
                print(
                    f"step={state.global_step} loss={sum(losses[-config.logging_steps:]) / len(losses[-config.logging_steps:]):.6f} lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if config.save_steps and state.global_step % config.save_steps == 0:
                save_training_checkpoint(
                    output / f"checkpoint-{state.global_step}",
                    model,
                    optimizer,
                    scheduler,
                    state,
                    model_path=model_path,
                    config=config,
                    metadata=checkpoint_metadata,
                )
                rotate_training_checkpoints(output, config.save_total_limit)
            if state.global_step >= planned_steps:
                stopped = True
                break
        if stopped:
            break
    checkpoint_path = save_training_checkpoint(
        output / f"checkpoint-{state.global_step}",
        model,
        optimizer,
        scheduler,
        state,
        model_path=model_path,
        config=config,
        metadata=checkpoint_metadata,
    )
    rotate_training_checkpoints(output, config.save_total_limit)
    final = output / "final"
    temporary = output / f".final.tmp-{uuid.uuid4().hex}"
    metrics = {
        "global_step": state.global_step,
        "mean_training_loss": sum(losses) / len(losses) if losses else None,
        "last_training_loss": losses[-1] if losses else None,
        "trainable_tt_parameters": summary.trainable_parameters,
        "total_model_parameters": summary.total_parameters,
        "tt_module_paths": list(summary.module_paths),
        "tt_backends": {
            path: module.backend_metadata() for path, module in modules.items()
        },
        "tt_activation_checkpointing": config.tt_activation_checkpointing,
        "model_gradient_checkpointing": config.gradient_checkpointing,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "resumed_from": str(resume_from) if resume_from else None,
        "resume_checkpoint": str(checkpoint_path),
    }
    try:
        export_tt_modules(
            model,
            temporary,
            model_path=model_path,
            purpose="tt-finetuned",
            core_dtype=torch.bfloat16,
            metadata={"global_step": state.global_step},
        )
        atomic_write_json(temporary / "training_metrics.json", metrics)
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metrics
