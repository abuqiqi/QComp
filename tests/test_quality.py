"""验证通用评测结果和 Causal LM 质量评测。

本模块使用返回确定 loss 的小型模型和已预处理 DataLoader，检查评测任务规范化、
动态 metrics、有效样本与 token 计数，以及模型训练状态恢复。

主要内容：
- ``_FixedLossCausalLM``：按照 batch 输入返回预设标量 loss。
- ``QualityEvaluationTests``：验证任务接口、质量指标和错误处理。
"""

import math
import unittest
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from qcomp.evaluation import EvaluationTask, evaluate_causal_lm


class _FixedLossCausalLM(nn.Module):
    """返回 batch 提供的标量 loss，并记录评测执行模式。"""

    def __init__(self) -> None:
        """初始化用于记录 forward 状态的列表。"""

        super().__init__()
        self.forward_states: list[tuple[bool, bool]] = []

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        loss_value: Tensor,
    ) -> dict[str, Tensor]:
        """返回调用者指定的标量 loss。

        参数：
            input_ids: 用于模拟 tokenized 输入的 token IDs。
            labels: 用于计算有效预测 token 数的目标 IDs。
            loss_value: 当前 batch 对应的预设平均 loss。

        返回：
            包含标量 loss 的模型输出映射。
        """

        del input_ids, labels
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        return {"loss": loss_value.reshape(())}


def _make_quality_loader() -> DataLoader[Any]:
    """创建具有不同有效预测 token 数的确定性 DataLoader。"""

    examples = (
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "labels": torch.tensor([0, 1, 2, -100]),
            "loss_value": torch.tensor(1.0),
        },
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "labels": torch.tensor([0, 3, -100, -100]),
            "loss_value": torch.tensor(3.0),
        },
    )
    return DataLoader(examples, batch_size=1, shuffle=False)


def _causal_lm_task(
    requested_metrics: tuple[str, ...] = ("loss", "perplexity"),
) -> EvaluationTask:
    """创建测试使用的 Causal LM 评测任务。

    参数：
        requested_metrics: 本次任务请求计算的指标名称。

    返回：
        使用固定数据集信息的评测任务。
    """

    return EvaluationTask(
        name="tiny-causal-lm",
        dataset="tiny-text",
        split="validation",
        preprocessing="causal-lm-blocks",
        requested_metrics=requested_metrics,
    )


class QualityEvaluationTests(unittest.TestCase):
    """验证通用任务接口和模型级 Causal LM 质量指标。"""

    def test_evaluate_causal_lm_returns_task_metrics(self) -> None:
        """按照有效 token 汇总动态指标并恢复模型训练状态。"""

        model = _FixedLossCausalLM()
        model.train()
        task = _causal_lm_task()

        result = evaluate_causal_lm(
            model,
            _make_quality_loader(),
            task,
            "cpu",
        )

        self.assertIs(result.task, task)
        self.assertAlmostEqual(result.metrics["loss"], 5.0 / 3.0)
        self.assertAlmostEqual(
            result.metrics["perplexity"],
            math.exp(5.0 / 3.0),
        )
        self.assertEqual(result.evaluated_examples, 2)
        self.assertEqual(result.evaluated_tokens, 3)
        self.assertEqual(model.forward_states, [(False, False), (False, False)])
        self.assertTrue(model.training)

    def test_evaluate_causal_lm_returns_only_requested_metrics(self) -> None:
        """只返回当前评测任务明确请求的指标。"""

        result = evaluate_causal_lm(
            _FixedLossCausalLM(),
            _make_quality_loader(),
            _causal_lm_task(("PERPLEXITY",)),
            "cpu",
        )

        self.assertEqual(tuple(result.metrics), ("perplexity",))

    def test_evaluate_causal_lm_rejects_unsupported_metric(self) -> None:
        """拒绝 Causal LM evaluator 无法计算的任务指标。"""

        with self.assertRaisesRegex(ValueError, "unsupported"):
            evaluate_causal_lm(
                _FixedLossCausalLM(),
                _make_quality_loader(),
                _causal_lm_task(("accuracy",)),
                "cpu",
            )

    def test_evaluate_causal_lm_requires_valid_prediction_tokens(self) -> None:
        """拒绝 labels 移位后没有有效预测 token 的数据。"""

        loader = DataLoader(
            (
                {
                    "input_ids": torch.tensor([0]),
                    "labels": torch.tensor([0]),
                    "loss_value": torch.tensor(1.0),
                },
            ),
            batch_size=1,
        )

        with self.assertRaisesRegex(ValueError, "no valid"):
            evaluate_causal_lm(
                _FixedLossCausalLM(),
                loader,
                _causal_lm_task(),
                "cpu",
            )
