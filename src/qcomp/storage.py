"""保存和加载通用张量网络 artifact。

本模块使用 PyTorch 存储机制序列化表示名称、格式版本、元数据和命名张量，并通过
``ArtifactPaths`` 配置数据集缓存、分解结果、checkpoint 和评测结果的标准目录。它
不导入 MPO 或任何计算后端，加载结果始终是通用 ``TensorNetworkArtifact``。

主要内容：
- ``ArtifactPaths``：根据一个可配置根目录提供标准实验产物路径。
- ``save_artifact``：把通用 artifact 写入指定路径。
- ``load_artifact``：从指定路径恢复并校验通用 artifact。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .representations.artifact import TensorNetworkArtifact


@dataclass(frozen=True)
class ArtifactPaths:
    """提供一个项目运行所使用的标准实验产物目录。"""

    root: str | Path = Path("artifacts")

    def __post_init__(self) -> None:
        """将 artifact 根目录规范为展开用户目录后的 Path。"""

        object.__setattr__(self, "root", Path(self.root).expanduser())

    @property
    def datasets(self) -> Path:
        """返回下载或预处理数据集的缓存目录。"""

        return self.root / "datasets"

    @property
    def cache(self) -> Path:
        """返回可重复生成的通用缓存目录。"""

        return self.root / "cache"

    @property
    def decompositions(self) -> Path:
        """返回张量网络分解结果目录。"""

        return self.root / "decompositions"

    @property
    def checkpoints(self) -> Path:
        """返回训练 checkpoint 目录。"""

        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        """返回评测结果和格式化报告目录。"""

        return self.root / "evaluations"

    def create_directories(self) -> None:
        """创建 artifact 根目录及全部标准子目录。"""

        for path in (
            self.datasets,
            self.cache,
            self.decompositions,
            self.checkpoints,
            self.evaluations,
        ):
            path.mkdir(parents=True, exist_ok=True)


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
