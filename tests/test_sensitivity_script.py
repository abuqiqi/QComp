"""验证 Qwen3 实验的参数校验、目标选择、workflow 编排与日志。

本模块使用小型模型和 mock 替代模型加载、backend 及评测，检查配置传递、目录和记录。

主要内容：
- ``SensitivityScriptTests``：验证实验入口与公共能力的连接。
"""

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import run_qwen3_sensitivity as experiment


class SensitivityScriptTests(unittest.TestCase):
    """验证实验配置及公共接口调用。"""

    def test_parse_args_uses_mmlu_defaults_and_accepts_other_tasks(self) -> None:
        """MMLU 使用默认指标，其他 task 显式指定指标。"""

        defaults = experiment.parse_args([])
        self.assertEqual(defaults.task, "mmlu")
        self.assertEqual(defaults.rank, 96)
        self.assertFalse(defaults.module_ranks)
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

    def test_rank_strategies(self) -> None:
        """验证分模块、统一和满秩策略及互斥参数。"""
        for argv, suffix, rank in [
            ([], "rank-96", 96),
            (["--module-ranks"], "module-ranks", 64),
            (["--rank", "32"], "rank-32", 32),
            (["--full-rank"], "full-rank", 256),
        ]:
            with (
                self.subTest(argv=argv),
                patch.object(experiment, "run_sensitivity_experiment") as run,
                patch.object(experiment, "plot_heatmaps"),
                patch.object(experiment, "capture_console", return_value=nullcontext()),
            ):
                experiment.main(argv)
                self.assertTrue(run.call_args.args[0].name.endswith(suffix))
                target = run.call_args.kwargs["make_target"](
                    "model.layers.0.self_attn.k_proj",
                    nn.Linear(2048, 1024, device="meta"),
                )
                self.assertEqual(target.spec.ranks, (1, rank, rank, 1))
                self.assertEqual(target.spec.in_modes, (16, 8, 16))
        for argv in [
            ["--rank", "32", "--full-rank"],
            ["--rank", "32", "--module-ranks"],
            ["--module-ranks", "--full-rank"],
        ]:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                experiment.parse_args(argv)
        with self.assertRaises(ValueError):
            experiment.make_qwen3_mpo_spec(nn.Linear(2048, 6144, device="meta"), 257)
        spec = experiment.make_qwen3_mpo_spec(
            nn.Linear(4096, 12288, device="meta"), None
        )
        self.assertEqual(spec.ranks, (1, 256, 768, 1))

    def test_layer_range_names_reject_legacy_cli(self) -> None:
        """明确区分 Linear 起点与题目起点，并拒绝旧参数和缩写。"""

        args = experiment.parse_args(["--start-layer-index", "3", "--max-layers", "2"])
        self.assertEqual((args.start_layer_index, args.max_layers), (3, 2))
        self.assertEqual(args.sample_start_index, 0)
        for option in ("--start-index", "--start-layer"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                experiment.parse_args([option, "1"])

    def test_invalid_parameters_fail_before_experiment(self) -> None:
        """非法评测参数、范围和 rank 在启动实验前失败。"""

        for option, value in (
            ("--batch-size", "0"),
            ("--max-length", "0"),
            ("--limit", "0"),
            ("--sample-start-index", "-1"),
            ("--num-fewshot", "-1"),
            ("--start-layer-index", "-1"),
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

        with (
            patch.object(experiment, "run_sensitivity_experiment") as run,
            patch.object(experiment, "plot_heatmaps") as plot,
            patch.object(experiment, "capture_console", return_value=nullcontext()),
        ):
            experiment.main(
                [
                    "--task",
                    "gsm8k",
                    "--metric",
                    "exact_match_strict_match",
                    "--model",
                    "custom/model",
                    "--sample-start-index",
                    "8000",
                    "--seed",
                    "17",
                    "--num-fewshot",
                    "2",
                    "--model-dtype",
                    "float32",
                    "--apply-chat-template",
                    "--rank",
                    "3",
                    "--start-layer-index",
                    "1",
                    "--max-layers",
                    "2",
                    "--output",
                    "out.md",
                    "--results",
                    "sensitivity_results.json",
                ]
            )
        plot.assert_called_once()
        self.assertEqual(
            plot.call_args.kwargs["report_path"], run.return_value.report_path
        )
        config = run.call_args.args[0]
        self.assertEqual(config.model.model_name_or_path, "custom/model")
        self.assertEqual(config.model.dtype, torch.float32)
        self.assertEqual(config.evaluation.evaluation.task, "gsm8k")
        self.assertEqual(config.evaluation.evaluation.num_fewshot, 2)
        self.assertEqual(config.evaluation.evaluation.evaluation_seed, 17)
        self.assertEqual(config.evaluation.evaluation.sample_start_index, 8000)
        self.assertTrue(config.evaluation.evaluation.apply_chat_template)
        self.assertEqual(
            tuple(config.evaluation.metric_directions), ("exact_match_strict_match",)
        )
        self.assertEqual((config.start_layer_index, config.max_layers), (1, 2))
        self.assertEqual(
            (config.output, config.results), ("out.md", "sensitivity_results.json")
        )
        linear = nn.Linear(4096, 1024, bias=False, device="meta")
        select = run.call_args.kwargs["select_linear"]
        self.assertTrue(select("block", linear))
        self.assertFalse(select("lm_head", linear))
        target = run.call_args.kwargs["make_target"]("block", linear)
        self.assertEqual(target.module_path, "block")
        self.assertEqual(target.representation, "mpo")
        self.assertEqual(target.spec.in_modes, (16, 16, 16))
        self.assertEqual(target.spec.out_modes, (16, 4, 16))
        self.assertEqual(target.spec.ranks, (1, 3, 3, 1))

    def test_default_paths_are_shared_before_model_loading(self) -> None:
        """自动路径在 workflow 前确定，完整覆盖评测和随后绘图。"""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(experiment, "run_sensitivity_experiment") as run,
            patch.object(experiment, "plot_heatmaps") as plot,
            patch.object(experiment, "capture_console") as capture,
        ):
            experiment.main(["--artifact-root", directory])
            config = run.call_args.args[0]
            report = Path(config.output)
            self.assertEqual(report.name, "report.md")
            self.assertRegex(report.parent.name, r"^\d{8}T\d{6}$")
            self.assertEqual(report.parent.parent.name, "qwen3-mmlu-mpo-rank-96")
            self.assertIsNone(config.results)
            capture.assert_called_once_with(report.parent / "console.log")
            capture.return_value.__enter__.assert_called_once()
            capture.return_value.__exit__.assert_called_once()
            run.assert_called_once()
            plot.assert_called_once()

    def test_heatmap_coordinates_units_and_missing_values(self) -> None:
        """乱序模块映射到真实块号，保留缺失格和带符号百分点。"""
        records = [
            {
                "layers": {"model.layers.4.mlp.down_proj": {}},
                "degradations": {"acc": 0.451},
            },
            {
                "layers": {"model.layers.2.self_attn.q_proj": {}},
                "degradations": {"acc": -0.02},
            },
        ]
        values, blocks, unit = experiment.heatmap_values(records, "acc")
        self.assertEqual(blocks, [2, 3, 4])
        self.assertEqual(unit, "pp")
        self.assertAlmostEqual(values[6, 2], 45.1)
        self.assertAlmostEqual(values[0, 0], -2)
        self.assertTrue(np.isnan(values[0, 1]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            experiment.heatmap_values(records + records, "acc")
        records[0]["degradations"] = {"word_perplexity": 2.5}
        values, _, unit = experiment.heatmap_values(records[:1], "word_perplexity")
        self.assertEqual(values[6, 0], 2.5)
        self.assertEqual(unit, "raw units")

    def test_heatmaps_show_each_baseline_in_metric_units(self) -> None:
        """多指标图分别显示基线，比例转百分比且原始指标保留数值。"""
        from PIL import Image

        metrics = [
            "exact_match_strict_match",
            "exact_match_flexible_extract",
            "word_perplexity",
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text(
                "- Evaluated examples: 256\n- Total evaluation examples: 1319\n"
                "\n## Baseline Metrics\n\n| Metric | Value |\n|---|---:|\n"
                "| exact_match_strict_match | 0.921875 |\n"
                "| exact_match_flexible_extract | 0.9296875 |\n"
                "| word_perplexity | 12.5 |\n"
            )
            records = [
                {
                    "layers": {"model.layers.0.self_attn.q_proj": {}},
                    "degradations": dict.fromkeys(metrics, 0.01),
                }
            ]
            images = experiment.plot_heatmaps(
                records,
                task="gsm8k",
                metrics=metrics,
                report_path=report,
            )
            for path, expected in zip(images, ["92.19%", "92.97%", "12.5"]):
                with Image.open(path) as image:
                    self.assertIn(f"Baseline: {expected}", image.info["Description"])
            report.write_text(report.read_text().replace("0.921875", "nan"))
            with self.assertRaisesRegex(ValueError, "baseline metric must be finite"):
                experiment.plot_heatmaps(
                    records,
                    task="gsm8k",
                    metrics=metrics,
                    report_path=report,
                )

    def test_plot_only_creates_png_and_updates_report_without_model(self) -> None:
        """补图从日志读取任务和指标，生成 PNG 并幂等嵌入报告，不运行模型。"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text(
                "# Existing report\n\n- Evaluated examples: 2\n- Total evaluation examples: 3270\n\n## Baseline Metrics\n\n| Metric | Value |\n|---|---:|\n| acc | 0.75 |\n"
            )
            from test_layer_selection_dashboard import make_sources

            data_config = make_sources(Path(directory), count=1, modules=1)
            log = Path(directory) / data_config["datasets"][0]["path"]
            data = json.loads(log.read_text())
            data["report_path"] = str(report)
            data["baseline"]["total_examples"] = 3270
            log.write_text(json.dumps(data))
            with patch.object(experiment, "run_sensitivity_experiment") as run:
                experiment.main(["--plot-only", str(log)])
                experiment.main(["--plot-only", str(log)])
                run.assert_not_called()
            png = report.with_name("report-acc-heatmap.png")
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            from PIL import Image

            with Image.open(png) as image:
                self.assertEqual(
                    image.info["Description"],
                    "Test samples: 2 / 3,270 (evaluation split) | Baseline: 75.00%",
                )
            self.assertEqual(report.read_text().count("![acc heatmap]"), 1)
            self.assertTrue(report.read_text().startswith("# Existing report"))
            data["expected_cases"].append("missing")
            log.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "case 不完整"):
                experiment.plot_results(log)


if __name__ == "__main__":
    unittest.main()
