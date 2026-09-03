"""公开可放入 PyTorch 模型的张量网络模块接口。

本包汇总通用线性模块协议和 MPO Linear 的共享实现，负责模型层的参数、输入输出形状
与 artifact 导出。具体计算库的分解和收缩逻辑仍由 backends 层提供。

主要内容：
- ``TensorNetworkLinear``：所有张量网络线性层共有的生命周期接口。
- ``MPOLinearBase``：统一 MPO Linear 的 artifact 解析和输入输出形状处理。
- ``CanonicalMPOLinearBase``：直接保存规范 MPO cores 的模型层基类。
"""

from .base import TensorNetworkLinear
from .mpo import CanonicalMPOLinearBase, MPOLinearBase

__all__ = [
    "CanonicalMPOLinearBase",
    "MPOLinearBase",
    "TensorNetworkLinear",
]
