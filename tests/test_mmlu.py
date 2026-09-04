"""验证 lm-eval MMLU 适配器的配置、结果转换和模型状态恢复。

本模块使用假的 lm-eval 模块执行确定性测试，不读取真实 MMLU 数据，也不运行大型
语言模型。测试确认 HFLM 参数、five-shot 配置、总体 accuracy、样本数和异常路径。

主要内容：
- `_install_fake_lm_eval`：在测试期间提供最小 lm-eval 模块结构。
- `MMLUEvaluationTests`：验证 MMLU 配置、调用参数、结果和状态恢复。
"""

import sys
import unittest
from types import ModuleType
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from qcomp.evaluation import MMLUEvaluationConfig, evaluate_mmlu


def _install_fake_lm_eval(
    simple_evaluate: Any,
    wrapper_calls: list[dict[str, Any]],
) -> dict[str, ModuleType]:
    """构造 evaluate_mmlu 导入所需的最小 lm-eval 模块。

    参数：
        simple_evaluate: 测试提供的评测函数。
        wrapper_calls: 保存 HFLM 构造参数的列表。

    返回：
        可以传给 `patch.dict(sys.modules)` 的模块映射。
    """

    class FakeHFLM:
        """记录 HFLM 构造参数的测试包装器。"""

        def __init__(self, **kwargs: Any) -> None:
            """保存本次包装器构造参数。

            参数：
                kwargs: evaluate_mmlu 传给 HFLM 的关键字参数。
            """

            wrapper_calls.append(kwargs)

    lm_eval = ModuleType("lm_eval")
    lm_eval.simple_evaluate = simple_evaluate
    models = ModuleType("lm_eval.models")
    huggingface = ModuleType("lm_eval.models.huggingface")
    huggingface.HFLM = FakeHFLM
    lm_eval.models = models
    models.huggingface = huggingface
    return {
        "lm_eval": lm_eval,
        "lm_eval.models": models,
        "lm_eval.models.huggingface": huggingface,
    }


class MMLUEvaluationTests(unittest.TestCase):
    """验证 MMLU 评测适配器的公共行为。"""

    def test_evaluate_mmlu_returns_unified_result(self) -> None:
        """传递 lm-eval 配置并返回总体 accuracy 和样本数。"""

        model = nn.Linear(2, 2, bias=False)
        model.train()
        tokenizer = object()
        wrapper_calls: list[dict[str, Any]] = []
        evaluation_calls: list[dict[str, Any]] = []

        def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
            """记录评测参数并返回确定的 MMLU 汇总。

            参数：
                kwargs: evaluate_mmlu 传给 lm-eval 的关键字参数。

            返回：
                包含总体 accuracy 和样本数的最小结果。
            """

            evaluation_calls.append(kwargs)
            self.assertFalse(model.training)
            return {
                "results": {
                    "mmlu": {
                        "acc,none": 0.625,
                        "sample_count": {"acc,none": 40},
                    }
                }
            }

        modules = _install_fake_lm_eval(simple_evaluate, wrapper_calls)
        config = MMLUEvaluationConfig(
            num_fewshot=5,
            batch_size=4,
            max_length=2048,
            limit=40,
            seed=17,
        )
        with patch.dict(sys.modules, modules):
            result = evaluate_mmlu(model, tokenizer, config)

        self.assertTrue(model.training)
        self.assertEqual(result.task, config.task)
        self.assertEqual(result.task.name, "mmlu")
        self.assertEqual(result.task.dataset, "cais/mmlu")
        self.assertEqual(result.task.split, "test")
        self.assertEqual(result.task.preprocessing, "lm-eval-5-shot")
        self.assertEqual(result.metrics, {"accuracy": 0.625})
        self.assertEqual(result.evaluated_examples, 40)
        self.assertIsNone(result.evaluated_tokens)
        self.assertEqual(
            wrapper_calls,
            [
                {
                    "pretrained": model,
                    "tokenizer": tokenizer,
                    "batch_size": 4,
                    "max_length": 2048,
                    "device": "cpu",
                }
            ],
        )
        call = evaluation_calls[0]
        self.assertEqual(call["tasks"], ["mmlu"])
        self.assertEqual(call["num_fewshot"], 5)
        self.assertEqual(call["limit"], 40)
        self.assertEqual(call["bootstrap_iters"], 0)
        self.assertFalse(call["apply_chat_template"])
        self.assertFalse(call["log_samples"])
        for seed_name in (
            "random_seed",
            "numpy_random_seed",
            "torch_random_seed",
            "fewshot_random_seed",
        ):
            self.assertEqual(call[seed_name], 17)

    def test_evaluate_mmlu_restores_state_after_failure(self) -> None:
        """lm-eval 失败时恢复模型原来的训练状态。"""

        model = nn.Linear(2, 2, bias=False)
        model.train()

        def fail_evaluation(**kwargs: Any) -> None:
            """模拟 lm-eval 执行失败。

            参数：
                kwargs: 未使用的 lm-eval 调用参数。
            """

            del kwargs
            raise RuntimeError("evaluation failed")

        modules = _install_fake_lm_eval(fail_evaluation, [])
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                evaluate_mmlu(model, object(), MMLUEvaluationConfig())
        self.assertTrue(model.training)

    def test_evaluate_mmlu_rejects_incomplete_result(self) -> None:
        """拒绝缺少 MMLU 汇总指标的 lm-eval 返回值。"""

        modules = _install_fake_lm_eval(lambda **kwargs: {"results": {}}, [])
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "accuracy"):
                evaluate_mmlu(
                    nn.Linear(2, 2, bias=False),
                    object(),
                    MMLUEvaluationConfig(),
                )

    def test_mmlu_config_validates_numeric_fields(self) -> None:
        """拒绝无效 few-shot、batch、上下文长度和样本限制。"""

        invalid_configs = (
            {"num_fewshot": -1},
            {"batch_size": 0},
            {"max_length": 0},
            {"limit": 0},
            {"limit": 1.0},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MMLUEvaluationConfig(**values)


if __name__ == "__main__":
    unittest.main()
