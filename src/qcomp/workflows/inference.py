"""执行完整 Causal LM 的正常文本生成并记录运行开销。

本模块接收已经预处理的 DataLoader，将 batch 张量移动到单个目标设备，并使用模型
自身的 generate 方法完成真实自回归生成。每次生成的新增 token 会移回 CPU，同时
汇总模型生成时间、workflow 总时间、token 吞吐和 PyTorch CUDA 峰值显存。数据读取、
文本解码、质量 metrics 和压缩率计算由其他层负责。

主要内容：
- ``InferenceConfig``：定义设备、最大生成长度、采样方式和随机种子。
- ``InferencePerformance``：保存逐 batch 时间、吞吐、token 数和峰值显存。
- ``InferenceResult``：组合 CPU 生成 token 与本次实际运行开销。
- ``infer_causal_lm``：逐 batch 调用模型 generate 并生成推理结果。
- ``_synchronize``：在 CUDA 计时边界同步设备工作。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class InferenceConfig:
    """定义单设备 Causal LM 正常生成配置。"""

    device: str = "cuda:0"
    max_new_tokens: int = 32
    do_sample: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """确认最大生成 token 数为正数。

        异常：
            ValueError: max_new_tokens 不是正数时抛出。
        """

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class InferencePerformance:
    """保存一次正常推理的实际时间、吞吐和峰值显存。"""

    batch_seconds: tuple[float, ...]
    total_seconds: float
    input_tokens: int
    generated_tokens: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    @property
    def generation_seconds(self) -> float:
        """返回全部 model.generate 调用的同步墙钟时间。"""

        return sum(self.batch_seconds)

    @property
    def generated_tokens_per_second(self) -> float:
        """返回 model.generate 时间内每秒生成的 token 数。"""

        return self.generated_tokens / self.generation_seconds


@dataclass(frozen=True)
class InferenceResult:
    """组合逐 batch 生成 token 与本次推理性能数据。"""

    generated_token_ids: tuple[Tensor, ...]
    performance: InferencePerformance


def _synchronize(device: torch.device) -> None:
    """等待指定 CUDA 设备上的异步工作完成。

    参数：
        device: 当前执行模型生成的设备。
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_causal_lm(
    model: nn.Module,
    dataloader: DataLoader[Any],
    config: InferenceConfig,
) -> InferenceResult:
    """使用模型自身的 generate 方法逐 batch 执行正常推理。

    参数：
        model: 提供 generate 方法的 decoder-only Causal LM。
        dataloader: 产生包含二维 input_ids 的已预处理 batch。
        config: 设备、最大生成长度、采样方式和随机种子。

    返回：
        新生成 token 的 CPU 张量和本次推理性能数据。

    异常：
        ValueError: batch、input_ids 或 generate 输出不符合接口约定时抛出。
        RuntimeError: 请求 CUDA 但当前环境没有 CUDA 时抛出。
        TypeError: 模型没有可调用的 generate 方法时抛出。
    """

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError("model must provide a callable generate method")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    original_training = model.training
    model.to(device)
    model.eval()
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    generated_batches: list[Tensor] = []
    batch_seconds: list[float] = []
    input_tokens = 0
    generated_tokens = 0
    total_start = perf_counter()
    try:
        with torch.inference_mode():
            for raw_batch in dataloader:
                if not isinstance(raw_batch, Mapping):
                    raise ValueError("dataloader batches must be mappings")
                input_ids = raw_batch.get("input_ids")
                if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                    raise ValueError("each batch must contain two-dimensional input_ids")
                model_inputs = {
                    name: value.to(device)
                    for name, value in raw_batch.items()
                    if name != "labels" and isinstance(value, Tensor)
                }
                device_input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs.get("attention_mask")
                if attention_mask is not None:
                    input_tokens += int(attention_mask.sum().detach().cpu().item())
                else:
                    input_tokens += device_input_ids.numel()

                _synchronize(device)
                batch_start = perf_counter()
                sequences = generate(
                    **model_inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                )
                _synchronize(device)
                batch_seconds.append(perf_counter() - batch_start)
                if not isinstance(sequences, Tensor) or sequences.ndim != 2:
                    raise ValueError("model.generate must return a two-dimensional Tensor")
                input_width = device_input_ids.shape[-1]
                if (
                    sequences.shape[0] != device_input_ids.shape[0]
                    or sequences.shape[-1] < input_width
                ):
                    raise ValueError(
                        "model.generate output must contain each input sequence"
                    )
                generated = sequences[:, input_width:].detach().cpu()
                generated_batches.append(generated)
                generated_tokens += generated.numel()
        _synchronize(device)
        total_seconds = perf_counter() - total_start
        if not generated_batches:
            raise ValueError("dataloader must contain at least one batch")
        peak_allocated_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        peak_reserved_bytes = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        )
    finally:
        model.train(original_training)

    return InferenceResult(
        generated_token_ids=tuple(generated_batches),
        performance=InferencePerformance(
            batch_seconds=tuple(batch_seconds),
            total_seconds=total_seconds,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
        ),
    )
