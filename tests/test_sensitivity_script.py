"""验证 Qwen3 实验的参数校验、目标选择、workflow 编排与日志。

本模块使用小型模型和 mock 替代模型加载、backend 及评测，检查配置传递、目录和记录。

主要内容：
- ``SensitivityScriptTests``：验证实验入口与公共能力的连接。
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts import run_qwen3_mmlu_sensitivity as experiment


class SensitivityScriptTests(unittest.TestCase):
    """验证实验配置及公共接口调用。"""

    def test_parse_args_uses_mmlu_defaults_and_accepts_other_tasks(self) -> None:
        """MMLU 使用默认指标，其他 task 显式指定指标。"""

        defaults = experiment.parse_args([])
        self.assertEqual(defaults.task, "mmlu")
        self.assertEqual(defaults.metric, ["acc"])
        self.assertIsNone(defaults.num_fewshot)
        gsm8k = experiment.parse_args(
            [
                "--task",
                "gsm8k",
                "--metric",
                "exact_match_strict_match",
                "--apply-chat-template",
            ]
        )
        self.assertEqual(gsm8k.task, "gsm8k")
        self.assertEqual(gsm8k.metric, ["exact_match_strict_match"])
        self.assertTrue(gsm8k.apply_chat_template)
        with self.assertRaises(SystemExit):
            experiment.parse_args(["--task", "gsm8k"])

    def test_invalid_parameters_fail_before_experiment(self) -> None:
        """非法评测参数、范围和 rank 在启动实验前失败。"""

        for option, value in (
            ("--batch-size", "0"),
            ("--max-length", "0"),
            ("--limit", "0"),
            ("--num-fewshot", "-1"),
            ("--start-index", "-1"),
            ("--max-layers", "0"),
            ("--rank", "0"),
        ):
            with (
                self.subTest(option=option),
                patch.object(experiment, "run_sensitivity_experiment") as run,
            ):
                with self.assertRaises(ValueError):
                    experiment.main([option, value])
                run.assert_not_called()

    def test_main_passes_config_and_qwen3_policies(self) -> None:
        """脚本向公共入口提供评测配置、选层规则与实际 MPO 目标构造。"""

        with patch.object(experiment, "run_sensitivity_experiment") as run:
            experiment.main(
                [
                    "--task",
                    "gsm8k",
                    "--metric",
                    "exact_match_strict_match",
                    "--model",
                    "custom/model",
                    "--seed",
                    "17",
                    "--num-fewshot",
                    "2",
                    "--model-dtype",
                    "float32",
                    "--apply-chat-template",
                    "--rank",
                    "3",
                    "--start-index",
                    "1",
                    "--max-layers",
                    "2",
                    "--output",
                    "out.md",
                    "--log",
                    "events.jsonl",
                ]
            )
        config = run.call_args.args[0]
        self.assertEqual(config.model.model_name_or_path, "custom/model")
        self.assertEqual(config.model.dtype, torch.float32)
        self.assertEqual(config.evaluation.task, "gsm8k")
        self.assertEqual(config.evaluation.num_fewshot, 2)
        self.assertEqual(config.evaluation.seed, 17)
        self.assertTrue(config.evaluation.apply_chat_template)
        self.assertEqual(config.metrics, ("exact_match_strict_match",))
        self.assertEqual((config.start_index, config.max_layers), (1, 2))
        self.assertEqual((config.output, config.log), ("out.md", "events.jsonl"))
        linear = nn.Linear(4096, 1024, bias=False, device="meta")
        select = run.call_args.kwargs["select_linear"]
        self.assertTrue(select("block", linear))
        self.assertFalse(select("lm_head", linear))
        target = run.call_args.kwargs["make_target"]("block", linear)
        self.assertEqual(target.module_path, "block")
        self.assertEqual(target.representation, "mpo")
        self.assertEqual(target.spec.in_modes, (16, 16, 16))
        self.assertEqual(target.spec.out_modes, (8, 8, 16))
        self.assertEqual(target.spec.ranks, (1, 3, 3, 1))


if __name__ == "__main__":
    unittest.main()
