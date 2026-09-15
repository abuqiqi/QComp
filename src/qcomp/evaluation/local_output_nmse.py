"""计算同一线性模块压缩前后的局部输出归一化平方误差。

本模块接收已经对齐的 reference 输出、压缩输出和有效 token mask，只负责逐批次的
误差/能量归约以及最终 NMSE 计算；模型执行、激活缓存和压缩编排由 workflows 层负责。

主要内容：
- ``local_output_error_sums``：计算一个 batch 的误差平方和、reference 能量和与有效 token 数。
- ``local_output_nmse``：由跨 batch 累计的分子和分母计算局部输出 NMSE。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def local_output_error_sums(
    reference_output: Tensor,
    compressed_output: Tensor,
    valid_mask: Tensor,
    *,
    valid_tokens: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """计算一个 batch 在有效 token 位置上的局部输出误差与能量。

    参数：
        reference_output: 原始模块输出，形状为 ``[batch, sequence, features]``。
        compressed_output: 压缩模块输出，形状和 reference 完全一致。
        valid_mask: 有效 token mask，形状为 ``[batch, sequence]``。
        valid_tokens: 同一 mask 的已知计数，避免逐候选设备同步。

    返回：
        FP64 标量形式的误差平方和、reference 输出能量和，以及有效 token 数。

    异常：
        ValueError: 输出或 mask 的形状不符合约定时抛出。
    """
    if reference_output.ndim != 3:
        raise ValueError("local outputs must have shape [batch, sequence, features]")
    if reference_output.shape != compressed_output.shape:
        raise ValueError("reference and compressed outputs must have identical shapes")
    expected_mask_shape = reference_output.shape[:2]
    if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(expected_mask_shape):
        raise ValueError("valid_mask must have shape [batch, sequence]")

    mask = valid_mask.to(device=reference_output.device, dtype=torch.bool)
    expanded_mask = mask.unsqueeze(-1)
    reference = reference_output.float()
    difference = compressed_output.float() - reference
    squared_error_sum = (
        difference.square().masked_fill(~expanded_mask, 0).sum(dtype=torch.float64)
    )
    target_power_sum = (
        reference.square().masked_fill(~expanded_mask, 0).sum(dtype=torch.float64)
    )
    return squared_error_sum, target_power_sum, int(mask.sum().item()) if valid_tokens is None else valid_tokens


def local_output_nmse(
    squared_error_sum: float | Tensor,
    target_power_sum: float | Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """由累计误差与 reference 能量计算局部输出 NMSE。

    参数：
        squared_error_sum: 全部 batch 的非负误差平方和。
        target_power_sum: 全部 batch 的非负 reference 输出能量和。
        epsilon: 加到分母的正数稳定项。

    返回：
        有限且非负的 Python 浮点 NMSE。

    异常：
        ValueError: 输入不是有限非负数或 epsilon 不是有限正数时抛出。
    """
    error = float(squared_error_sum)
    power = float(target_power_sum)
    if not math.isfinite(error) or error < 0:
        raise ValueError("squared_error_sum must be finite and non-negative")
    if not math.isfinite(power) or power < 0:
        raise ValueError("target_power_sum must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    result = error / (power + epsilon)
    if not math.isfinite(result):
        raise ValueError("local output NMSE must be finite")
    return result
