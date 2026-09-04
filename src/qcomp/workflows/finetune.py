"""编排只更新张量网络参数的 Causal LM 微调。

本模块查找模型中已经安装且启用梯度的 ``TensorNetworkLinear`` 参数，使用默认
Causal LM objective 调用 training 层的公共训练循环，并导出每个压缩层训练后的
canonical artifact。训练循环和 checkpoint 不在这里重复实现。

主要内容：
- ``TensorNetworkFineTuneResult``：组合公共训练结果与张量网络特有产物。
- ``finetune_tensor_network_causal_lm``：执行张量网络参数专用的微调流程。
- ``_tensor_network_parameter_names``：取得允许公共训练循环更新的参数名称。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from torch import nn
from torch.utils.data import DataLoader

from ..model import list_tensor_network_linears
from ..representations import TensorNetworkArtifact
from ..training import (
    CausalLMObjective,
    TrainingConfig,
    TrainingResult,
    train_causal_lm,
)


@dataclass(frozen=True)
class TensorNetworkFineTuneResult:
    """组合通用训练结果与张量网络微调产生的特有产物。"""

    training: TrainingResult
    module_paths: tuple[str, ...]
    tn_artifacts: Mapping[str, TensorNetworkArtifact]


def _tensor_network_parameter_names(
    model: nn.Module,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回张量网络模块路径和已启用梯度的参数名称。

    参数：
        model: 已经安装张量网络 Linear 的模型。

    返回：
        张量网络模块路径和允许训练的完整参数名称。

    异常：
        ValueError: 模型没有张量网络层或没有启用梯度的张量网络参数时抛出。
    """

    modules = list_tensor_network_linears(model)
    if not modules:
        raise ValueError("model contains no TensorNetworkLinear modules")
    selected_ids = {
        id(parameter)
        for _, module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    parameter_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in selected_ids
    )
    if not parameter_names:
        raise ValueError("model contains no trainable tensor-network parameters")
    return tuple(path for path, _ in modules), parameter_names


def finetune_tensor_network_causal_lm(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> TensorNetworkFineTuneResult:
    """只更新模型中的张量网络参数并导出最终 artifacts。

    参数：
        model: 已经安装可训练 ``TensorNetworkLinear`` 的 Causal LM。
        train_dataloader: 产生已 tokenize batch 的可重复迭代 DataLoader。
        config: 单设备训练配置。
        output_dir: checkpoint 输出目录。
        resume_from: 可选的明确 checkpoint 文件路径。

    返回：
        通用训练结果、张量网络模块路径和最终 artifacts。
    """

    module_paths, parameter_names = _tensor_network_parameter_names(model)
    training = train_causal_lm(
        model,
        train_dataloader,
        parameter_names,
        CausalLMObjective(),
        config,
        output_dir,
        resume_from=resume_from,
    )
    artifacts = {
        path: module.export_artifact()
        for path, module in list_tensor_network_linears(model)
    }
    return TensorNetworkFineTuneResult(
        training=training,
        module_paths=module_paths,
        tn_artifacts=artifacts,
    )
