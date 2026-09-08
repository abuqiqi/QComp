"""验证通用 lm-eval evaluator 的任务复用和结果转换。

本模块用假的 lm-eval 模块覆盖 task、group、动态指标、模型状态恢复和参数校验；测试
只读取项目 runtime 配置，不读取真实 benchmark，也不访问网络。

主要内容：
- ``_install_fake_lm_eval``：提供最小 HFLM、TaskManager 和 simple_evaluate。
- ``LMEvalEvaluatorTests``：验证任意 task/group 共用一个 evaluator 实现。
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from torch import nn

from qcomp.evaluation import LMEvalConfig, LMEvalEvaluator


def _install_fake_lm_eval(
    simple_evaluate: Any,
    *,
    loaded: dict[str, Any],
    wrapper_calls: list[dict[str, Any]],
    load_calls: list[list[str]],
) -> dict[str, ModuleType]:
    """构造 evaluator 延迟导入所需的最小 lm-eval 模块树。"""

    class FakeHFLM:
        """记录 Hugging Face 模型包装参数。"""

        def __init__(self, **kwargs: Any) -> None:
            """保存包装参数。"""

            wrapper_calls.append(kwargs)

    class FakeTaskManager:
        """返回调用方指定的 task/group 对象。"""

        def load(self, tasks: list[str]) -> dict[str, Any]:
            """记录任务名称并返回预置结果。"""

            load_calls.append(tasks)
            return loaded

    lm_eval = ModuleType("lm_eval")
    lm_eval.simple_evaluate = simple_evaluate
    tasks = ModuleType("lm_eval.tasks")
    tasks.TaskManager = FakeTaskManager
    models = ModuleType("lm_eval.models")
    huggingface = ModuleType("lm_eval.models.huggingface")
    huggingface.HFLM = FakeHFLM
    lm_eval.models = models
    models.huggingface = huggingface
    return {
        "lm_eval": lm_eval,
        "lm_eval.tasks": tasks,
        "lm_eval.models": models,
        "lm_eval.models.huggingface": huggingface,
    }


class LMEvalEvaluatorTests(unittest.TestCase):
    """验证一个 evaluator 可执行任意 lm-eval task 或 group。"""

    def test_group_returns_dynamic_metrics_and_reuses_task(self) -> None:
        """group 动态提取指标，并在重复模型状态评测时只加载一次。"""

        model = nn.Linear(2, 2, bias=False)
        model.train()
        tokenizer = object()
        group = object()
        wrapper_calls: list[dict[str, Any]] = []
        load_calls: list[list[str]] = []
        evaluation_calls: list[dict[str, Any]] = []

        def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
            """记录调用并返回 group 指标。"""

            evaluation_calls.append(kwargs)
            self.assertFalse(model.training)
            return {
                "groups": {
                    "mmlu": {
                        "acc,none": 0.625,
                        "acc_stderr,none": 0.02,
                        "sample_len": 40,
                    }
                }
            }

        modules = _install_fake_lm_eval(
            simple_evaluate,
            loaded={"groups": {"mmlu": group}, "tasks": {}},
            wrapper_calls=wrapper_calls,
            load_calls=load_calls,
        )
        config = LMEvalConfig(
            task="mmlu",
            num_fewshot=5,
            batch_size=4,
            max_length=2048,
            limit=40,
            seed=17,
        )
        with patch.dict(sys.modules, modules):
            evaluator = LMEvalEvaluator(tokenizer, config)
            first = evaluator(model)
            second = evaluator(model)

        self.assertTrue(model.training)
        self.assertEqual(first, second)
        self.assertEqual(first.metrics, {"acc": 0.625})
        self.assertEqual(first.evaluated_examples, 40)
        self.assertEqual(first.task.name, "mmlu")
        self.assertEqual(first.task.preprocessing, "lm-eval-5-shot")
        self.assertEqual(first.task.requested_metrics, ("acc",))
        self.assertEqual(load_calls, [["mmlu"]])
        self.assertEqual(len(wrapper_calls), 2)
        self.assertEqual(wrapper_calls[0]["pretrained"], model)
        self.assertEqual(wrapper_calls[0]["tokenizer"], tokenizer)
        self.assertEqual(wrapper_calls[0]["device"], "cpu")
        self.assertEqual(len(evaluation_calls), 2)
        self.assertIs(evaluation_calls[0]["tasks"][0], group)
        self.assertIs(
            evaluation_calls[0]["task_manager"],
            evaluation_calls[1]["task_manager"],
        )
        self.assertEqual(evaluation_calls[0]["num_fewshot"], 5)
        self.assertEqual(evaluation_calls[0]["limit"], 40)
        self.assertTrue(evaluation_calls[0]["cache_requests"])
        for seed_name in (
            "random_seed",
            "numpy_random_seed",
            "torch_random_seed",
            "fewshot_random_seed",
        ):
            self.assertEqual(evaluation_calls[0][seed_name], 17)

    def test_standalone_task_uses_filtered_metric_and_n_samples(self) -> None:
        """单 task 从 results 和 n-samples 提取带 filter 的指标。"""

        task = object()
        modules = _install_fake_lm_eval(
            lambda **kwargs: {
                "results": {"gsm8k": {"exact_match,strict-match": 0.75}},
                "n-samples": {"gsm8k": {"effective": 12}},
            },
            loaded={"groups": {}, "tasks": {"gsm8k": task}},
            wrapper_calls=[],
            load_calls=[],
        )
        with patch.dict(sys.modules, modules):
            result = LMEvalEvaluator(
                object(),
                LMEvalConfig(task="gsm8k", num_fewshot=None),
            )(nn.Linear(2, 2, bias=False))

        self.assertEqual(result.metrics, {"exact_match_strict_match": 0.75})
        self.assertEqual(result.evaluated_examples, 12)
        self.assertEqual(result.task.preprocessing, "lm-eval-default-shot")

    def test_sample_ranges_do_not_overlap_and_repeat_consistently(self) -> None:
        """前缀与后续范围覆盖所有题目，重复评测沿用相同索引并绕过缓存。"""
        task = SimpleNamespace(eval_docs=list(range(10)))
        calls = []

        def evaluate(**kwargs: Any) -> dict[str, Any]:
            """记录选择的题目，返回实际评测数量。"""
            indices = kwargs["samples"]
            chosen = indices["boolq"] if indices else task.eval_docs[:kwargs["limit"]]
            calls.append((list(chosen), kwargs))
            return {
                "results": {"boolq": {"acc,none": 0.5}},
                "n-samples": {"boolq": {"effective": len(chosen)}},
            }

        modules = _install_fake_lm_eval(
            evaluate, loaded={"tasks": {"boolq": task}},
            wrapper_calls=[], load_calls=[],
        )
        model = nn.Linear(2, 2, bias=False)
        with patch.dict(sys.modules, modules):
            LMEvalEvaluator(object(), LMEvalConfig(task="boolq", limit=6))(model)
            tail = LMEvalEvaluator(
                object(), LMEvalConfig(task="boolq", sample_start_index=6)
            )
            result = tail(model)
            tail(model)
            bounded = LMEvalEvaluator(
                object(), LMEvalConfig(task="boolq", sample_start_index=6, limit=2)
            )(model)
            clipped = LMEvalEvaluator(
                object(), LMEvalConfig(task="boolq", sample_start_index=6, limit=100)
            )(model)
        self.assertEqual(calls[0][0] + calls[1][0], list(range(10)))
        self.assertEqual(calls[1][0], calls[2][0])
        self.assertEqual(calls[3][0], [6, 7])
        for _, kwargs in calls[1:]:
            self.assertIsNone(kwargs["limit"])
            self.assertFalse(kwargs["cache_requests"])
        self.assertEqual(result.evaluated_examples, 4)
        self.assertEqual(bounded.evaluated_examples, 2)
        self.assertEqual(clipped.evaluated_examples, 4)
        self.assertIn("samples-6:10", result.task.preprocessing)

    def test_sample_range_rejects_groups_and_out_of_bounds(self) -> None:
        """拒绝 group 偏移及空范围，避免空索引触发全量评测。"""
        for loaded, start, message in (
            ({"groups": {"boolq": object()}}, 1, "single task"),
            ({"tasks": {"boolq": SimpleNamespace(eval_docs=[1, 2])}}, 2, "outside"),
        ):
            evaluate = Mock()
            modules = _install_fake_lm_eval(
                evaluate, loaded=loaded, wrapper_calls=[], load_calls=[],
            )
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, message):
                    LMEvalEvaluator(
                        object(), LMEvalConfig(task="boolq", sample_start_index=start)
                    )(nn.Linear(2, 2, bias=False))
            evaluate.assert_not_called()

    def test_failure_restores_model_state(self) -> None:
        """lm-eval 抛出异常时仍恢复模型原来的训练状态。"""

        model = nn.Linear(2, 2, bias=False)
        model.train()
        modules = _install_fake_lm_eval(
            Mock(side_effect=RuntimeError("evaluation failed")),
            loaded={"groups": {}, "tasks": {"boolq": object()}},
            wrapper_calls=[],
            load_calls=[],
        )
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                LMEvalEvaluator(object(), LMEvalConfig(task="boolq"))(model)
        self.assertTrue(model.training)

    def test_config_rejects_invalid_values(self) -> None:
        """拒绝空 task、无效数值和非布尔 chat template 开关。"""

        invalid = (
            {"task": ""},
            {"task": "x", "num_fewshot": -1},
            {"task": "x", "batch_size": 0},
            {"task": "x", "max_length": 0},
            {"task": "x", "limit": 0},
            {"task": "x", "limit": 1.0},
            {"task": "x", "sample_start_index": -1},
            {"task": "x", "sample_start_index": True},
            {"task": "x", "sample_start_index": 1.5},
            {"task": "x", "sample_start_index": 1, "limit": 0.5},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    LMEvalConfig(**values)
        with self.assertRaisesRegex(TypeError, "apply_chat_template"):
            LMEvalConfig(task="x", apply_chat_template="yes")


if __name__ == "__main__":
    unittest.main()
