"""Configuration and state types for TT-only training."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TTFineTuneConfig:
    num_train_epochs: int = 1
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    device: str = "cuda:0"
    bf16_autocast: bool = True
    gradient_checkpointing: bool = False
    tt_activation_checkpointing: bool = True
    first_step_peak_memory_limit_gib: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_grad_norm",
            "max_length",
            "logging_steps",
            "save_total_limit",
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
        if (
            self.first_step_peak_memory_limit_gib is not None
            and self.first_step_peak_memory_limit_gib <= 0
        ):
            raise ValueError("first_step_peak_memory_limit_gib must be positive")


@dataclass(frozen=True)
class TTTrainingState:
    global_step: int = 0
    epoch: int = 0
    batch_index: int = 0
    tokens_seen: int = 0


@dataclass(frozen=True)
class TTTrainableSummary:
    module_paths: tuple[str, ...]
    parameter_tensors: int
    trainable_parameters: int
    total_parameters: int
