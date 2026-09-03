"""使用同步后的墙钟时间测量张量网络后端操作。

本模块统一执行预热、重复测量和 CUDA 同步，为分解、无 bias 推理及完整 SGD 训练
step 返回同一种计时结果。后端构造和资源释放遵循各公开函数定义的计时边界。

主要内容：
- ``TimingResult``：保存原始耗时样本，并提供平均值和中位数。
- ``time_decomposition``：测量 backend 分解稠密权重的耗时。
- ``time_inference``：测量已构造模型层的 forward 耗时。
- ``time_training_step``：测量清梯度、forward、backward 和参数更新的总耗时。
- ``_synchronize``、``_measure``：提供内部设备同步和通用计时流程。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from typing import Any, TypeVar

import torch
from torch import Tensor

from ..backends import TensorNetworkBackend
from ..representations import TensorNetworkArtifact

SpecT = TypeVar("SpecT")


@dataclass(frozen=True)
class TimingResult:
    """以秒为单位保存多次墙钟时间测量结果。"""

    samples_seconds: tuple[float, ...]

    @property
    def mean_seconds(self) -> float:
        """返回全部样本的算术平均值。"""

        return mean(self.samples_seconds)

    @property
    def median_seconds(self) -> float:
        """返回全部样本的中位数。"""

        return median(self.samples_seconds)


def _synchronize(device: torch.device) -> None:
    """等待计时设备上排队的 CUDA 工作完成。

    参数：
        device: 被测操作使用的设备。
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    operation: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimingResult:
    """完成不计时预热后测量可调用对象。

    参数：
        operation: 待执行的无参数操作。
        device: 需要同步异步工作的设备。
        warmup: 不计时的预热次数。
        repeats: 正式计时的执行次数。

    返回：
        各次同步后的墙钟时间样本。
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    for _ in range(warmup):
        operation()
    _synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = perf_counter()
        operation()
        _synchronize(device)
        samples.append(perf_counter() - start)
    return TimingResult(tuple(samples))


def time_decomposition(
    backend: TensorNetworkBackend[SpecT],
    weight: Tensor,
    spec: SpecT,
    *,
    warmup: int = 1,
    repeats: int = 5,
) -> TimingResult:
    """测量后端对单个稠密矩阵执行原生分解的时间。

    参数：
        backend: 提供分解能力的后端。
        weight: 每次调用都要分解的稠密矩阵。
        spec: 与当前张量网络表示对应的分解配置。
        warmup: 不计时的分解次数。
        repeats: 正式计时的分解次数。

    返回：
        同步后的分解计时结果。
    """

    if not backend.capabilities.decomposition:
        raise ValueError(
            f"{backend.provider}/{backend.representation} does not support decomposition"
        )

    def operation() -> None:
        """执行一次后端分解。"""

        backend.decompose(weight, spec)

    return _measure(
        operation,
        device=weight.device,
        warmup=warmup,
        repeats=repeats,
    )


def time_inference(
    backend: TensorNetworkBackend[Any],
    tn_artifact: TensorNetworkArtifact,
    inputs: Tensor,
    *,
    warmup: int = 2,
    repeats: int = 10,
) -> TimingResult:
    """使用共享的张量网络 artifact 测量前向推理时间。

    参数：
        backend: 提供推理运行时的后端。
        tn_artifact: 各运行时比较时共享的张量网络 artifact。
        inputs: 无 bias Linear 的输入。
        warmup: 不计时的前向调用次数。
        repeats: 正式计时的前向调用次数。

    返回：
        同步后的推理计时结果。
    """

    if not backend.capabilities.inference:
        raise ValueError(
            f"{backend.provider}/{backend.representation} does not support inference"
        )
    runtime_tn_artifact = tn_artifact.to(inputs.device, inputs.dtype)
    linear = backend.build_linear(runtime_tn_artifact, trainable=False)
    try:
        linear.eval()

        def operation() -> None:
            """执行一次推理前向调用。"""

            with torch.inference_mode():
                linear(inputs)

        return _measure(
            operation,
            device=inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
    finally:
        linear.close()


def time_training_step(
    backend: TensorNetworkBackend[Any],
    tn_artifact: TensorNetworkArtifact,
    inputs: Tensor,
    targets: Tensor,
    *,
    warmup: int = 1,
    repeats: int = 5,
    learning_rate: float = 1e-3,
) -> TimingResult:
    """测量可训练张量网络运行时的完整 SGD steps。

    参数：
        backend: 提供可训练运行时的后端。
        tn_artifact: 各运行时比较时共享的张量网络 artifact。
        inputs: 无 bias Linear 的输入。
        targets: 与运行时输出匹配的 MSE 目标值。
        warmup: 不计时的训练 steps 数量。
        repeats: 正式计时的训练 steps 数量。
        learning_rate: 所有被比较后端统一使用的 SGD 学习率。

    返回：
        同步后的训练 step 计时结果。
    """

    if not backend.capabilities.training:
        raise ValueError(
            f"{backend.provider}/{backend.representation} does not support training"
        )
    runtime_tn_artifact = tn_artifact.to(inputs.device, inputs.dtype)
    linear = backend.build_linear(runtime_tn_artifact, trainable=True)
    try:
        linear.train()
        optimizer = torch.optim.SGD(linear.parameters(), lr=learning_rate)

        def operation() -> None:
            """依次执行梯度清零、前向、MSE、反向传播和优化器更新。"""

            optimizer.zero_grad(set_to_none=True)
            prediction = linear(inputs)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()

        return _measure(
            operation,
            device=inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
    finally:
        linear.close()
