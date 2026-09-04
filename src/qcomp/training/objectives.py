"""定义可插拔的 Causal LM 训练目标。

本模块规定公共训练循环如何调用 loss objective，并提供直接读取模型 ``output.loss``
的默认实现。objective 只负责前向与 loss 计算，不选择可训练参数，也不执行反向传播、
优化器更新或 checkpoint。

主要内容：
- ``TrainingObjective``：约定 objective 的持久化元数据和 loss 调用接口。
- ``CausalLMObjective``：调用 ``model(**batch)`` 并读取返回结果中的 loss。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn


class TrainingObjective(Protocol):
    """定义公共训练循环接受的 loss objective 接口。"""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """返回用于 checkpoint 一致性检查的可持久化配置。"""

        ...

    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
    ) -> Tensor:
        """根据模型和一个 batch 计算标量 loss。

        参数：
            model: 当前训练的模型。
            batch: 已经移动到训练设备的数据。

        返回：
            用于反向传播的标量 loss。
        """

        ...


@dataclass(frozen=True)
class CausalLMObjective:
    """使用 Causal LM 自身返回的监督训练 loss。"""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """返回默认 Causal LM objective 的稳定标识。"""

        return {"name": "causal_lm"}

    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
    ) -> Tensor:
        """执行模型并取得输出中的 loss。

        参数：
            model: 当前训练的 Causal LM。
            batch: 传给模型的已 tokenize batch。

        返回：
            模型输出中的 loss 张量。

        异常：
            ValueError: 模型输出没有提供 Tensor 类型的 loss 时抛出。
        """

        output = model(**batch)
        loss = (
            output.get("loss")
            if isinstance(output, Mapping)
            else getattr(output, "loss", None)
        )
        if not isinstance(loss, Tensor):
            raise ValueError("causal LM output must contain a Tensor loss")
        return loss
