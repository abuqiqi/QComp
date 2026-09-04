"""测量张量网络表示的压缩规模和重建质量。

本模块既可以比较单个稠密权重与通用张量网络 artifact，也可以根据模型级压缩结果
统计完整模型压缩前后的参数量和张量字节数。表示相关的稠密重建由 representations
层统一分派；模型级统计使用 canonical artifacts，不依赖执行 backend 的内部布局。

主要内容：
- ``CompressionMetrics``：保存单个张量的规模、压缩率和重建误差。
- ``ModelCompressionMetrics``：保存完整模型压缩前后的规模和压缩率。
- ``compression_metrics``：重建任意已注册表示并生成单张量指标。
- ``model_compression_metrics``：根据模型压缩结果生成完整模型指标。
- ``_tensor_size``：统一统计张量元素数和数据字节数。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..representations import TensorNetworkArtifact, reconstruct_tensor

if TYPE_CHECKING:
    from ..workflows.compress import ModelCompressionResult


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


@dataclass(frozen=True)
class ModelCompressionMetrics:
    """汇总完整模型压缩前后的参数量、张量字节数和压缩率。"""

    dense_parameters: int
    compressed_parameters: int
    compression_ratio: float
    dense_tensor_bytes: int
    compressed_tensor_bytes: int
    tensor_size_compression_ratio: float
    compressed_layers: int


def _tensor_size(tensors: Iterable[Tensor]) -> tuple[int, int]:
    """统计一组张量的元素总数和数据字节总数。

    参数：
        tensors: 待统计的张量迭代器。

    返回：
        张量元素总数和数据字节总数。
    """

    parameters = 0
    tensor_bytes = 0
    for tensor in tensors:
        parameters += tensor.numel()
        tensor_bytes += tensor.numel() * tensor.element_size()
    return parameters, tensor_bytes


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
    dense_parameters, dense_tensor_bytes = _tensor_size((weight,))
    compressed_parameters, compressed_tensor_bytes = _tensor_size(
        tn_artifact.tensors.values()
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


def model_compression_metrics(
    model: nn.Module,
    compression_result: ModelCompressionResult,
) -> ModelCompressionMetrics:
    """计算模型级压缩前后的参数量和张量字节数。

    当前模型必须仍安装 ``compression_result`` 记录的压缩层。未压缩参数直接从模型
    统计；目标层的压缩规模使用 canonical artifact，而不使用执行 backend 的内部参数。

    参数：
        model: 当前安装了模型级压缩结果的完整 PyTorch 模型。
        compression_result: ``compress_model()`` 返回的模型级压缩结果。

    返回：
        完整模型压缩前后的参数量、张量字节数、压缩率和压缩层数量。

    异常：
        ValueError: 结果记录的压缩层已不在对应模型路径时抛出。
    """

    replacement_parameter_ids: set[int] = set()
    for layer_result in compression_result.layer_results:
        replacement = layer_result.replacement
        if model.get_submodule(replacement.target) is not replacement.replacement:
            raise ValueError(
                f"compressed layer is not installed at {replacement.target!r}"
            )
        replacement_parameter_ids.update(
            id(parameter) for parameter in replacement.replacement.parameters()
        )

    unchanged_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if id(parameter) not in replacement_parameter_ids
    )
    unchanged_parameter_count, unchanged_tensor_bytes = _tensor_size(
        unchanged_parameters
    )
    dense_target_parameters, dense_target_bytes = _tensor_size(
        parameter
        for layer_result in compression_result.layer_results
        for parameter in layer_result.replacement.original.parameters()
    )
    compressed_target_parameters, compressed_target_bytes = _tensor_size(
        tensor
        for layer_result in compression_result.layer_results
        for tensor in layer_result.tn_artifact.tensors.values()
    )

    dense_parameters = unchanged_parameter_count + dense_target_parameters
    compressed_parameters = (
        unchanged_parameter_count + compressed_target_parameters
    )
    dense_tensor_bytes = unchanged_tensor_bytes + dense_target_bytes
    compressed_tensor_bytes = unchanged_tensor_bytes + compressed_target_bytes
    return ModelCompressionMetrics(
        dense_parameters=dense_parameters,
        compressed_parameters=compressed_parameters,
        compression_ratio=dense_parameters / compressed_parameters,
        dense_tensor_bytes=dense_tensor_bytes,
        compressed_tensor_bytes=compressed_tensor_bytes,
        tensor_size_compression_ratio=(
            dense_tensor_bytes / compressed_tensor_bytes
        ),
        compressed_layers=len(compression_result.layer_results),
    )
