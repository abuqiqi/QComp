"""评测 Causal LM 的平均 loss 和 perplexity。

本模块接收评测任务和已经预处理的 DataLoader，把 batch 张量移动到指定设备，复用
训练层的 CausalLMObjective 执行完整模型前向，并按照有效预测 token 数汇总任务请求
的 loss 与 perplexity。数据集读取和 tokenize 属于 data 层，不在这里实现。

主要内容：
- ``evaluate_causal_lm``：执行模型级 Causal LM 质量评测并返回通用结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from math import exp
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..training import CausalLMObjective
from .task import EvaluationResult, EvaluationTask

_CAUSAL_LM_METRICS = frozenset(("loss", "perplexity"))


def evaluate_causal_lm(
    model: nn.Module,
    dataloader: DataLoader[Any],
    task: EvaluationTask,
    device: str | torch.device,
) -> EvaluationResult:
    """计算评测任务请求的 Causal LM loss 和 perplexity。

    模型返回的 loss 应当是移位后非 -100 labels 的平均负对数似然，这与 Hugging Face
    Causal LM 的 loss 约定一致。

    参数：
        model: 接收 tokenized batch 并返回标量 loss 的 Causal LM。
        dataloader: 产生包含 labels 的已预处理 batch 的 DataLoader。
        task: 指定数据集信息和所需 metrics 的评测任务。
        device: 执行完整模型前向的设备。

    返回：
        包含任务信息、动态指标、样本数和有效预测 token 数的通用结果。

    异常：
        ValueError: metric、batch、labels 或模型 loss 不符合评测约定时抛出。
        RuntimeError: 请求 CUDA 但当前环境没有 CUDA 时抛出。
    """

    unsupported_metrics = tuple(
        metric for metric in task.requested_metrics if metric not in _CAUSAL_LM_METRICS
    )
    if unsupported_metrics:
        raise ValueError(
            f"unsupported causal LM metrics: {unsupported_metrics}"
        )
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    objective = CausalLMObjective()
    weighted_loss = 0.0
    evaluated_examples = 0
    evaluated_tokens = 0
    original_training = model.training
    model.to(runtime_device)
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in dataloader:
                if not isinstance(raw_batch, Mapping):
                    raise ValueError("dataloader batches must be mappings")
                batch = {
                    name: value.to(runtime_device)
                    if isinstance(value, Tensor)
                    else value
                    for name, value in raw_batch.items()
                }
                labels = batch.get("labels")
                if not isinstance(labels, Tensor) or labels.ndim == 0:
                    raise ValueError("each batch must contain Tensor labels")
                valid_labels = labels[..., 1:].ne(-100)
                token_count = int(valid_labels.sum().detach().cpu().item())
                if token_count == 0:
                    continue
                loss = objective(model, batch)
                if loss.numel() != 1 or not torch.isfinite(loss):
                    raise ValueError(
                        "causal LM evaluation loss must be finite and scalar"
                    )
                weighted_loss += float(loss.detach().cpu()) * token_count
                evaluated_examples += (
                    int(valid_labels.any(dim=-1).sum().detach().cpu().item())
                    if labels.ndim > 1
                    else 1
                )
                evaluated_tokens += token_count
    finally:
        model.train(original_training)

    if evaluated_tokens == 0:
        raise ValueError("dataloader contains no valid causal LM prediction tokens")
    mean_loss = weighted_loss / evaluated_tokens
    available_metrics = {
        "loss": mean_loss,
        "perplexity": exp(mean_loss),
    }
    return EvaluationResult(
        task=task,
        metrics={
            metric: available_metrics[metric]
            for metric in task.requested_metrics
        },
        evaluated_examples=evaluated_examples,
        evaluated_tokens=evaluated_tokens,
    )
