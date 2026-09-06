"""加载 Causal LM，并列出、查找、替换和恢复其 Linear 层。

本模块通过 Transformers 通用接口从本地路径或模型 ID 加载完整 Causal LM 和
tokenizer，并提供模型结构操作。Transformers 仅在调用加载函数时导入；张量分解、
计算后端、数据预处理、训练和评测仍由相邻模块负责。

主要内容：
- ``ModelLoadConfig``：定义模型来源、目标设备、数据类型和远程代码选项。
- ``CausalLMResources``：组合加载后的模型与 tokenizer。
- ``load_causal_lm``：使用 Transformers 通用接口加载单设备 Causal LM。
- ``LinearReplacement``：记录目标路径、原始 Linear 和压缩模型层。
- ``list_linears``：列出模型中可由当前压缩层替换的 Linear。
- ``list_tensor_network_linears``：列出模型中已经安装的张量网络 Linear。
- ``find_linear``：按模块路径查找并验证无 bias Linear。
- ``replace_linear``：安装已经构造好的张量网络 Linear。
- ``restore_linear``：关闭压缩模型层并恢复原始 Linear。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from .nn import TensorNetworkLinear
from .runtime import configure_runtime

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelLoadConfig:
    """定义 Hugging Face Causal LM 的单设备加载配置。"""

    model_name_or_path: str | None = None
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        """确认模型名称或路径不是空字符串。

        异常：
            ValueError: 模型名称或路径为空时抛出。
        """

        if self.model_name_or_path is not None and not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")


@dataclass(frozen=True)
class CausalLMResources:
    """组合已经加载到目标设备的 Causal LM 与 tokenizer。"""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase


def load_causal_lm(
    config: ModelLoadConfig | None = None,
    *,
    runtime_config_path: str | Path | None = None,
) -> CausalLMResources:
    """通过 Transformers 通用接口加载单设备 Causal LM 和 tokenizer。

    参数：
        config: 可选模型来源、目标设备、数据类型和远程代码配置。
        runtime_config_path: 可选 runtime TOML；省略时读取项目默认配置。

    返回：
        已经移动到目标设备的模型及其 tokenizer。

    异常：
        RuntimeError: 请求 CUDA 但当前环境没有 CUDA 时抛出。
        ModuleNotFoundError: 当前环境没有安装 Transformers 时抛出。
    """

    runtime = configure_runtime(runtime_config_path)
    load_config = config or ModelLoadConfig()
    device = torch.device(load_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA model loading requested but CUDA is unavailable")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = load_config.model_name_or_path or runtime.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=load_config.trust_remote_code,
        local_files_only=runtime.offline,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=load_config.dtype,
        trust_remote_code=load_config.trust_remote_code,
        local_files_only=runtime.offline,
    )
    model.to(device)
    return CausalLMResources(model=model, tokenizer=tokenizer)


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


def list_tensor_network_linears(
    model: nn.Module,
) -> tuple[tuple[str, TensorNetworkLinear], ...]:
    """列出模型中已经安装的张量网络 Linear。

    参数：
        model: 待遍历的 PyTorch 模型。

    返回：
        按 ``named_modules()`` 顺序排列的模块路径和张量网络 Linear 对。
    """

    return tuple(
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, TensorNetworkLinear)
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
