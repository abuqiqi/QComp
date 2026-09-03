"""在 PyTorch 模型中列出、查找、替换和恢复无 bias Linear 层。

本模块只处理模型结构，不执行张量分解，也不选择计算后端。调用者先通过 backend
构造 ``TensorNetworkLinear``，再按模块路径安装到模型中；返回的替换记录用于恢复
原始 ``torch.nn.Linear``。

主要内容：
- ``LinearReplacement``：记录目标路径、原始 Linear 和压缩模型层。
- ``list_linears``：列出模型中可由当前压缩层替换的 Linear。
- ``find_linear``：按模块路径查找并验证无 bias Linear。
- ``replace_linear``：安装已经构造好的张量网络 Linear。
- ``restore_linear``：关闭压缩模型层并恢复原始 Linear。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .nn import TensorNetworkLinear


@dataclass(frozen=True)
class LinearReplacement:
    """保存一次模型层替换所需的恢复信息。"""

    target: str
    original: nn.Linear
    replacement: TensorNetworkLinear


def list_linears(model: nn.Module) -> tuple[tuple[str, nn.Linear], ...]:
    """列出模型中具有模块路径的无 bias Linear。

    参数：
        model: 待遍历的 PyTorch 模型。

    返回：
        按 ``named_modules()`` 顺序排列的模块路径和 Linear 对。
    """

    return tuple(
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and module.bias is None
    )


def _parent_module(model: nn.Module, target: str) -> tuple[nn.Module, str]:
    """返回目标模块的父模块和属性名。

    参数：
        model: 包含目标层的 PyTorch 模型。
        target: 相对于模型根节点的模块路径。

    返回：
        目标层的父模块和最后一级属性名。

    异常：
        ValueError: 目标路径为空时抛出。
    """

    if not target:
        raise ValueError("target must not be empty")
    parent_path, _, name = target.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, name


def find_linear(model: nn.Module, target: str) -> nn.Linear:
    """按模块路径查找一个无 bias Linear。

    参数：
        model: 包含目标层的 PyTorch 模型。
        target: 例如 ``layers.0.self_attn.q_proj`` 的模块路径。

    返回：
        找到的无 bias ``torch.nn.Linear``。

    异常：
        TypeError: 目标模块不是 ``torch.nn.Linear`` 时抛出。
        ValueError: 目标 Linear 包含 bias 时抛出。
    """

    module = model.get_submodule(target)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"target is not torch.nn.Linear: {target!r}")
    if module.bias is not None:
        raise ValueError(f"target Linear must not contain bias: {target!r}")
    return module


def replace_linear(
    model: nn.Module,
    target: str,
    replacement: TensorNetworkLinear,
) -> LinearReplacement:
    """将一个无 bias Linear 替换为已构造的张量网络模型层。

    参数：
        model: 包含目标层的 PyTorch 模型。
        target: 相对于模型根节点的模块路径。
        replacement: backend 已经构造好的张量网络 Linear。

    返回：
        后续恢复原始层所需的替换记录。
    """

    original = find_linear(model, target)
    parent, name = _parent_module(model, target)
    setattr(parent, name, replacement)
    return LinearReplacement(target, original, replacement)


def restore_linear(model: nn.Module, record: LinearReplacement) -> None:
    """关闭压缩模型层并恢复替换前的 Linear。

    参数：
        model: 当前安装了压缩层的 PyTorch 模型。
        record: ``replace_linear`` 返回的替换记录。
    """

    parent, name = _parent_module(model, record.target)
    record.replacement.close()
    setattr(parent, name, record.original)
