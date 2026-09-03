"""定义与具体表示无关的张量网络数据容器。

本模块保存表示名称、格式版本、结构元数据和独立命名的张量，不假设数据一定包含
cores 或 ranks。它负责 artifact 自身的基础校验和设备、数据类型转换，不解释具体
张量网络结构。

主要内容：
- ``TensorNetworkArtifact``：通用不可变 artifact，并提供命名张量查询与转换方法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorNetworkArtifact:
    """保存命名张量和元数据，但不假设具体张量网络结构。"""

    representation: str
    metadata: Mapping[str, Any]
    tensors: Mapping[str, Tensor]
    format_version: int = 1

    def __post_init__(self) -> None:
        """规范化映射并校验持久化 artifact 的边界。"""

        representation = self.representation.strip().lower()
        if not representation:
            raise ValueError("representation must not be empty")
        if self.format_version != 1:
            raise ValueError(
                f"unsupported artifact format version: {self.format_version}"
            )
        tensors = dict(self.tensors)
        if not tensors:
            raise ValueError("artifact must contain at least one tensor")
        if any(not isinstance(tensor, Tensor) for tensor in tensors.values()):
            raise TypeError("artifact tensors must be torch.Tensor values")
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "tensors", tensors)

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "TensorNetworkArtifact":
        """将全部张量移动到指定设备和可选数据类型。

        参数：
            device: ``torch.Tensor.to`` 接受的目标设备。
            dtype: 可选的目标浮点数据类型。

        返回：
            包含转换后张量的新 artifact。
        """

        tensors = {
            name: tensor.to(device=device, dtype=dtype or tensor.dtype)
            for name, tensor in self.tensors.items()
        }
        return TensorNetworkArtifact(
            representation=self.representation,
            format_version=self.format_version,
            metadata=self.metadata,
            tensors=tensors,
        )
