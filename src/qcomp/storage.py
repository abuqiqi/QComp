"""保存和加载通用张量网络 artifact。

本模块使用 PyTorch 存储机制序列化表示名称、格式版本、元数据和命名张量。它不导入
MPO 或任何计算后端，加载结果始终是通用 ``TensorNetworkArtifact``。

主要内容：
- ``save_artifact``：把通用 artifact 写入指定路径。
- ``load_artifact``：从指定路径恢复并校验通用 artifact。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .representations.artifact import TensorNetworkArtifact


def save_artifact(
    path: str | Path,
    artifact: TensorNetworkArtifact,
) -> None:
    """保存不包含后端特有对象的张量网络 artifact。

    参数：
        path: 目标文件路径。
        artifact: 待保存的通用 artifact。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "representation": artifact.representation,
            "format_version": artifact.format_version,
            "metadata": dict(artifact.metadata),
            "tensors": dict(artifact.tensors),
        },
        destination,
    )


def load_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> TensorNetworkArtifact:
    """加载与具体表示无关的张量网络 artifact。

    参数：
        path: Artifact 文件路径。
        map_location: 传给 ``torch.load`` 的可选目标位置。

    返回：
        加载得到的通用 artifact。
    """

    payload: dict[str, Any] = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    return TensorNetworkArtifact(
        representation=payload["representation"],
        format_version=payload["format_version"],
        metadata=payload["metadata"],
        tensors=payload["tensors"],
    )
