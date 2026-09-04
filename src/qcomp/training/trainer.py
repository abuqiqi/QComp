"""执行与具体微调方法无关的 Causal LM 训练循环。

本模块接收模型、可训练参数名称、loss objective 和 DataLoader，统一执行单设备训练、
梯度累积、学习率调度及断点续训。参数选择和方法特有产物导出由 workflows 负责；
objective 的具体 loss 计算由 objectives 模块负责。

主要内容：
- ``TrainingConfig``：定义训练步数、优化器、梯度和 checkpoint 配置。
- ``TrainingResult``：返回通用训练状态、损失和 checkpoint 路径。
- ``train_causal_lm``：使用指定参数和 objective 执行公共训练流程。
- ``_save_checkpoint``、``_load_checkpoint``：保存和恢复通用续训状态。
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .objectives import TrainingObjective


@dataclass(frozen=True)
class TrainingConfig:
    """定义单设备 Causal LM 训练配置。"""

    num_train_epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    save_steps: int = 0
    seed: int = 42
    device: str = "cuda:0"
    bf16_autocast: bool = True

    def __post_init__(self) -> None:
        """校验会影响训练循环和调度器的配置值。"""

        for name in (
            "num_train_epochs",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_grad_norm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.save_steps < 0:
            raise ValueError("save_steps must be non-negative")


@dataclass(frozen=True)
class TrainingResult:
    """保存通用训练循环产生的状态、损失和 checkpoint。"""

    global_step: int
    losses: tuple[float, ...]
    trainable_parameters: int
    parameter_names: tuple[str, ...]
    checkpoint_path: Path


@dataclass(frozen=True)
class _TrainingState:
    """记录下一个待处理 batch 的位置和已完成的优化步数。"""

    global_step: int = 0
    epoch: int = 0
    batch_index: int = 0


def _build_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    """创建先线性预热、再线性衰减的学习率调度器。

    参数：
        optimizer: 待调节学习率的优化器。
        total_steps: 完整训练计划包含的优化步数。
        warmup_ratio: 用于预热的训练步数比例。

    返回：
        与完整训练计划对应的 PyTorch 调度器。
    """

    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        """返回指定优化步使用的学习率比例。"""

        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(
            0.0,
            (total_steps - step) / max(1, total_steps - warmup_steps),
        )

    return LambdaLR(optimizer, scale)


def _capture_rng_state() -> dict[str, Any]:
    """捕获 Python、PyTorch 和全部 CUDA 设备的随机数状态。"""

    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复 checkpoint 中保存的随机数状态。

    参数：
        state: ``_capture_rng_state`` 生成的状态映射。
    """

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_state_to_device(optimizer: Optimizer) -> None:
    """把恢复的优化器张量移动到对应参数所在设备。

    参数：
        optimizer: 已经加载状态的优化器。
    """

    for parameter, values in optimizer.state.items():
        for name, value in values.items():
            if isinstance(value, Tensor):
                values[name] = value.to(parameter.device)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    state: _TrainingState,
    config: TrainingConfig,
    parameter_names: tuple[str, ...],
    objective_metadata: Mapping[str, Any],
    total_steps: int,
    batches_per_epoch: int,
) -> Path:
    """保存继续训练所需的最小状态。

    参数：
        path: checkpoint 文件路径。
        model: 当前训练模型。
        optimizer: 当前优化器。
        scheduler: 当前学习率调度器。
        state: 当前训练循环位置。
        config: 本次训练配置。
        parameter_names: 被优化参数的完整名称。
        objective_metadata: 当前 loss objective 的可持久化说明。
        total_steps: 完整训练计划的优化步数。
        batches_per_epoch: DataLoader 每个 epoch 的 batch 数量。

    返回：
        已写入的 checkpoint 路径。
    """

    parameters = dict(model.named_parameters())
    payload = {
        "format": "qcomp-training-checkpoint-v1",
        "config": asdict(config),
        "state": asdict(state),
        "parameter_names": parameter_names,
        "objective": dict(objective_metadata),
        "total_steps": total_steps,
        "batches_per_epoch": batches_per_epoch,
        "parameters": {
            name: parameters[name].detach().cpu() for name in parameter_names
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    config: TrainingConfig,
    parameter_names: tuple[str, ...],
    objective_metadata: Mapping[str, Any],
    total_steps: int,
    batches_per_epoch: int,
) -> _TrainingState:
    """恢复并校验同一压缩模型的续训状态。

    参数：
        path: 待加载的 checkpoint 文件。
        model: 具有相同参数结构的当前训练模型。
        optimizer: 待恢复的优化器。
        scheduler: 待恢复的学习率调度器。
        config: 当前训练配置。
        parameter_names: 当前被优化参数名称。
        objective_metadata: 当前 loss objective 的可持久化说明。
        total_steps: 当前完整训练计划步数。
        batches_per_epoch: 当前 DataLoader 的 batch 数量。

    返回：
        checkpoint 保存的训练循环位置。

    异常：
        ValueError: checkpoint 与当前训练任务不匹配时抛出。
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": "qcomp-training-checkpoint-v1",
        "config": asdict(config),
        "parameter_names": parameter_names,
        "objective": dict(objective_metadata),
        "total_steps": total_steps,
        "batches_per_epoch": batches_per_epoch,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"checkpoint {name} does not match current training")

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["parameters"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_state_to_device(optimizer)
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload["rng"])
    return _TrainingState(**payload["state"])


def train_causal_lm(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    trainable_parameter_names: Sequence[str],
    objective: TrainingObjective,
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """使用指定参数和 objective 训练 Causal LM。

    参数：
        model: 待训练的 Causal LM。
        train_dataloader: 产生已 tokenize batch 的可重复迭代 DataLoader。
        trainable_parameter_names: 唯一允许更新的模型参数全名。
        objective: 根据模型和 batch 返回标量 loss 的训练目标。
        config: 单设备训练配置。
        output_dir: checkpoint 输出目录。
        resume_from: 可选的明确 checkpoint 文件路径。

    返回：
        通用训练状态、本次调用的损失和最新 checkpoint。

    异常：
        ValueError: DataLoader 为空、参数名称无效或 objective 返回无效 loss 时抛出。
        RuntimeError: 请求 CUDA 但当前环境没有 CUDA 时抛出。
    """

    batches_per_epoch = len(train_dataloader)
    if batches_per_epoch == 0:
        raise ValueError("train_dataloader must not be empty")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model.to(device)
    parameter_names = tuple(trainable_parameter_names)
    if not parameter_names:
        raise ValueError("trainable_parameter_names must not be empty")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("trainable_parameter_names must not contain duplicates")
    named_parameters = dict(model.named_parameters())
    unknown = tuple(name for name in parameter_names if name not in named_parameters)
    if unknown:
        raise ValueError(f"unknown trainable parameters: {unknown}")

    model.requires_grad_(False)
    parameters = tuple(named_parameters[name] for name in parameter_names)
    for parameter in parameters:
        parameter.requires_grad_(True)
    objective_metadata = dict(objective.metadata)
    optimizer = AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / config.gradient_accumulation_steps
    )
    total_steps = config.max_steps or config.num_train_epochs * updates_per_epoch
    total_epochs = (
        math.ceil(total_steps / updates_per_epoch)
        if config.max_steps is not None
        else config.num_train_epochs
    )
    scheduler = _build_scheduler(optimizer, total_steps, config.warmup_ratio)
    state = _TrainingState()
    if resume_from is not None:
        state = _load_checkpoint(
            Path(resume_from),
            model,
            optimizer,
            scheduler,
            config,
            parameter_names,
            objective_metadata,
            total_steps,
            batches_per_epoch,
        )

    output = Path(output_dir)
    losses: list[float] = []
    last_checkpoint: Path | None = None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stopped = state.global_step >= total_steps
    sampler = getattr(train_dataloader, "sampler", None)
    for epoch in range(state.epoch, total_epochs):
        if stopped:
            break
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
                config.gradient_accumulation_steps,
                batches_per_epoch - window_start,
            )
            batch = {
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in raw_batch.items()
            }
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and config.bf16_autocast
                else nullcontext()
            )
            with autocast:
                loss = objective(model, batch)
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError("training objective must return a scalar Tensor")
            if not torch.isfinite(loss):
                raise ValueError("training objective loss must be finite")
            losses.append(float(loss.detach().cpu()))
            (loss / window_size).backward()
            if (
                (batch_index + 1) % config.gradient_accumulation_steps
                and batch_index + 1 != batches_per_epoch
            ):
                continue

            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch == batches_per_epoch:
                next_epoch = epoch + 1
                next_batch = 0
            state = _TrainingState(state.global_step + 1, next_epoch, next_batch)
            if config.save_steps and state.global_step % config.save_steps == 0:
                last_checkpoint = _save_checkpoint(
                    output / f"checkpoint-{state.global_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    state,
                    config,
                    parameter_names,
                    objective_metadata,
                    total_steps,
                    batches_per_epoch,
                )
            if state.global_step >= total_steps:
                stopped = True
                break

    final_path = output / f"checkpoint-{state.global_step}.pt"
    if last_checkpoint != final_path:
        last_checkpoint = _save_checkpoint(
            final_path,
            model,
            optimizer,
            scheduler,
            state,
            config,
            parameter_names,
            objective_metadata,
            total_steps,
            batches_per_epoch,
        )
    return TrainingResult(
        global_step=state.global_step,
        losses=tuple(losses),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        parameter_names=parameter_names,
        checkpoint_path=final_path,
    )
