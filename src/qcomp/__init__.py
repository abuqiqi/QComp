"""公开 qcomp 第一版的顶层接口。

本包集中导出 artifact、通用重建入口、MPO 表示工具、后端查询、模型层替换和持久化
函数，为调用者提供稳定入口。具体 Provider 保持隔离，只有通过 registry 选择后才会
被加载。

主要内容：
- ``TensorNetworkArtifact``：保存与具体张量网络表示无关的数据。
- ``reconstruct_tensor``：根据 artifact 的表示类型重建稠密张量。
- ``MPOSpec``：描述 MPO 的输入维度、输出维度和 TT ranks。
- ``get_backend``、``list_backends``：查询 Provider 与表示的已注册组合。
- ``list_linears``、``find_linear``、``replace_linear``、``restore_linear``：操作模型中的 Linear。
- ``make_mpo_artifact``、``parse_mpo_artifact``、``reconstruct_mpo``：转换和重建 MPO。
- ``save_artifact``、``load_artifact``：保存和加载通用 artifact。
"""

from .backends import get_backend, list_backends
from .model import (
    LinearReplacement,
    find_linear,
    list_linears,
    replace_linear,
    restore_linear,
)
from .representations import (
    MPOSpec,
    TensorNetworkArtifact,
    make_mpo_artifact,
    parse_mpo_artifact,
    reconstruct_mpo,
    reconstruct_tensor,
)
from .storage import load_artifact, save_artifact

__all__ = [
    "MPOSpec",
    "LinearReplacement",
    "TensorNetworkArtifact",
    "find_linear",
    "get_backend",
    "list_backends",
    "list_linears",
    "load_artifact",
    "make_mpo_artifact",
    "parse_mpo_artifact",
    "reconstruct_mpo",
    "reconstruct_tensor",
    "replace_linear",
    "restore_linear",
    "save_artifact",
]
