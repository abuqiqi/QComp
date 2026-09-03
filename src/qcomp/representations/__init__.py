"""公开张量网络表示的数据结构和数学操作。

本包汇总通用 artifact、表示级重建分派与规范 MPO 接口，只负责数据格式、结构校验和
数学变换，不选择具体计算后端，也不实现 PyTorch 模型层。

主要内容：
- ``TensorNetworkArtifact``：保存任意张量网络表示的元数据和命名张量。
- ``reconstruct_tensor``：根据 artifact 的表示类型调用对应稠密重建函数。
- ``MPOSpec``：描述规范 MPO 的维度与 ranks。
- ``make_mpo_artifact``、``parse_mpo_artifact``：转换 MPO 与通用 artifact。
- ``reconstruct_mpo``：从规范 cores 重建稠密权重。
"""

from .artifact import TensorNetworkArtifact
from .mpo import (
    MPOSpec,
    make_mpo_artifact,
    parse_mpo_artifact,
    reconstruct_mpo,
)
from .registry import reconstruct_tensor

__all__ = [
    "MPOSpec",
    "TensorNetworkArtifact",
    "make_mpo_artifact",
    "parse_mpo_artifact",
    "reconstruct_mpo",
    "reconstruct_tensor",
]
