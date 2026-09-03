"""测量张量网络表示的压缩规模和重建质量。

本模块比较稠密参考权重与通用张量网络 artifact，计算参数量、张量字节数、两类
压缩率和相对重建误差。表示相关的稠密重建由 representations 层统一分派，本模块
不依赖具体表示或计算 backend。

主要内容：
- ``CompressionMetrics``：保存参数量、张量字节数、压缩率和重建误差。
- ``compression_metrics``：重建任意已注册表示并生成压缩指标。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..representations import TensorNetworkArtifact, reconstruct_tensor


@dataclass(frozen=True)
class CompressionMetrics:
    """汇总参数与张量字节缩减程度及稠密权重重建质量。"""

    dense_parameters: int
    compressed_parameters: int
    compression_ratio: float
    dense_tensor_bytes: int
    compressed_tensor_bytes: int
    tensor_size_compression_ratio: float
    relative_error: float


def compression_metrics(
    weight: Tensor,
    tn_artifact: TensorNetworkArtifact,
) -> CompressionMetrics:
    """测量参数量和相对重建误差。

    参数：
        weight: 稠密参考矩阵。
        tn_artifact: 待评测的通用张量网络 artifact。

    返回：
        压缩和重建指标。
    """

    reconstructed = reconstruct_tensor(tn_artifact)
    if weight.shape != reconstructed.shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match reconstructed "
            f"shape {tuple(reconstructed.shape)}"
        )
    dense_parameters = weight.numel()
    compressed_parameters = sum(
        tensor.numel() for tensor in tn_artifact.tensors.values()
    )
    dense_tensor_bytes = weight.numel() * weight.element_size()
    compressed_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in tn_artifact.tensors.values()
    )
    relative_error = (
        torch.linalg.vector_norm(reconstructed - weight)
        / torch.linalg.vector_norm(weight)
    ).item()
    return CompressionMetrics(
        dense_parameters=dense_parameters,
        compressed_parameters=compressed_parameters,
        compression_ratio=dense_parameters / compressed_parameters,
        dense_tensor_bytes=dense_tensor_bytes,
        compressed_tensor_bytes=compressed_tensor_bytes,
        tensor_size_compression_ratio=(
            dense_tensor_bytes / compressed_tensor_bytes
        ),
        relative_error=relative_error,
    )
