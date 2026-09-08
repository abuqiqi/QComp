"""验证 Qwen3 联合压缩脚本的层范围、MPO 微调、产物与断点续训。

本模块用确定性小模型替代 Qwen3 加载，并模拟数据来源与 lm-eval；分解、替换、训练、
checkpoint 和 artifact 读写均使用真实公共接口，检查多层共同生效及未压缩参数冻结。

主要内容：
- ``TinyQwen``：提供 Qwen3 模块路径和可训练 loss 的小模型。
- ``tiny_spec``：为测试中的四维投影构造小型 MPO。
- ``AlpacaFineTuneScriptTests``：验证选层、三阶段评测、续训和失败恢复。
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from qcomp import MPOSpec, list_tensor_network_linears, load_artifact
from scripts import run_alpaca_finetune as experiment


class TinyQwen(nn.Module):
    """保留编号 24、25、26 的 attention 与 MLP 投影路径。"""

    def __init__(self, width: int = 4, device: str = "cpu") -> None:
        """构建给定特征宽度 width、设备 device 上的确定性模型。"""

        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(4, width, device=device)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(24)])
        for _ in range(3):
            block = nn.Module()
            for group, names in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                container = nn.Module()
                for name in names:
                    container.add_module(
                        name, nn.Linear(width, width, bias=False, device=device)
                    )
                block.add_module(group, container)
            block.other_proj = nn.Linear(width, width, bias=False, device=device)
            self.model.layers.append(block)
        self.lm_head = nn.Linear(width, 4, bias=False, device=device)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """根据输入 input_ids 与目标 labels 计算标量训练 loss。"""

        hidden = self.embedding(input_ids)
        for block in self.model.layers[24:]:
            projections = tuple(block.self_attn.children()) + tuple(
                block.mlp.children()
            )
            hidden = hidden + sum(layer(hidden) for layer in projections) / len(
                projections
            )
        logits = self.lm_head(hidden)
        return {
            "loss": nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        }


def tiny_spec(linear: nn.Linear, rank: int) -> MPOSpec:
    """为四维 linear 构造内部 bond 为 rank 的双核 MPO。"""

    assert linear.in_features == linear.out_features == 4
    return MPOSpec(out_modes=(2, 2), in_modes=(2, 2), ranks=(1, rank, 1))


class AlpacaFineTuneScriptTests(unittest.TestCase):
    """覆盖 block 范围边界、联合压缩、参数冻结与训练恢复。"""

    def test_default_range_selects_all_seven_projections_from_25(self) -> None:
        """默认包含编号 25 和末层，排除编号 24、输出头及非目标 proj。"""

        args = experiment.parse_args([])
        self.assertEqual((args.start_layer, args.end_layer, args.rank), (25, None, 96))
        self.assertIsNone(args.max_blocks)
        model = TinyQwen(width=4096, device="meta")
        plan = experiment.make_compression_plan(
            model, args.start_layer, args.end_layer, args.rank
        )
        self.assertEqual(len(plan.targets), 14)
        paths = [target.module_path for target in plan.targets]
        self.assertTrue(
            all(
                path.startswith(("model.layers.25.", "model.layers.26."))
                for path in paths
            )
        )
        self.assertTrue(
            all(target.spec.ranks == (1, 96, 96, 1) for target in plan.targets)
        )
        bounded = experiment.make_compression_plan(model, 25, 26, 96)
        self.assertEqual(len(bounded.targets), 7)
        self.assertTrue(
            all(
                target.module_path.startswith("model.layers.25.")
                for target in bounded.targets
            )
        )
        for start, stop in ((27, None), (25, 28), (26, 25)):
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                experiment.make_compression_plan(model, start, stop, 96)

    def test_bad_configuration_fails_before_model_loading(self) -> None:
        """非法范围、rank、训练和评测参数在模型加载之前失败。"""

        for argv in (
            ["--rank", "0"],
            ["--start-layer", "-1"],
            ["--end-layer", "25"],
            ["--batch-size", "0"],
            ["--num-train-epochs", "0"],
            ["--eval-limit", "0"],
        ):
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                patch.object(experiment, "load_causal_lm") as load,
                self.assertRaises((SystemExit, ValueError)),
            ):
                experiment.main(argv)
            load.assert_not_called()

    def run_experiment(
        self,
        root: Path,
        resume_from: Path | None = None,
        fail_evaluation: bool = False,
    ) -> tuple[TinyQwen, list[dict[str, object]]]:
        """在 root 执行真实小模型训练，可从 resume_from 续训或模拟压缩后评测失败。"""

        model = TinyQwen()
        snapshots = []
        records = [
            {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 0])},
        ]
        loader = DataLoader(records, batch_size=1)

        def evaluate(current: nn.Module) -> SimpleNamespace:
            """记录 current 模型的压缩层与参数，返回固定评测结果。"""

            snapshots.append(
                {
                    "paths": tuple(
                        path for path, _ in list_tensor_network_linears(current)
                    ),
                    "parameters": {
                        name: value.detach().clone()
                        for name, value in current.named_parameters()
                    },
                    "trainable": tuple(
                        name
                        for name, value in current.named_parameters()
                        if value.requires_grad
                    ),
                }
            )
            if fail_evaluation and len(snapshots) == 2:
                raise RuntimeError("模拟联合压缩后评测失败")
            return SimpleNamespace(metrics={"acc": 0.5}, evaluated_examples=2)

        argv = [
            "--device",
            "cpu",
            "--model-dtype",
            "float32",
            "--rank",
            "2",
            "--decomposition-provider",
            "native",
            "--execution-provider",
            "native",
            "--grad-accum-steps",
            "1",
            "--save-steps",
            "1",
            "--learning-rate",
            "0.01",
            "--artifact-root",
            str(root),
            "--eval-limit",
            "2",
        ]
        if resume_from is not None:
            argv.extend(["--resume-from", str(resume_from)])
        with (
            redirect_stdout(io.StringIO()),
            patch.object(
                experiment,
                "load_causal_lm",
                return_value=SimpleNamespace(model=model, tokenizer=object()),
            ),
            patch.object(experiment, "make_qwen3_mpo_spec", side_effect=tiny_spec),
            patch.object(experiment, "build_causal_lm_dataloader", return_value=loader),
            patch.object(experiment, "EVAL_TASKS", (("mmlu", 5, "acc"),)),
            patch.object(experiment, "LMEvalEvaluator", return_value=evaluate),
            patch.object(
                experiment, "compress_model", wraps=experiment.compress_model
            ) as compress,
        ):
            if fail_evaluation:
                with self.assertRaisesRegex(RuntimeError, "模拟"):
                    experiment.main(argv)
            else:
                experiment.main(argv)
            self.assertEqual(
                compress.call_args.kwargs["decomposition_dtype"], torch.float32
            )
        return model, snapshots

    def test_joint_compression_training_and_distinct_artifacts(self) -> None:
        """三阶段依次评测，所有目标同时替换，训练仅更新 MPO 并分别保存产物。"""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, snapshots = self.run_experiment(root)
            self.assertEqual([len(stage["paths"]) for stage in snapshots], [0, 14, 14])
            before, after = snapshots[1]["parameters"], snapshots[2]["parameters"]
            trained_names = snapshots[2]["trainable"]
            self.assertTrue(
                all(name.startswith(snapshots[2]["paths"]) for name in trained_names)
            )
            self.assertTrue(
                any(
                    not torch.equal(before[name], after[name]) for name in trained_names
                )
            )
            for name in before.keys() - set(trained_names):
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
            self.assertEqual(list_tensor_network_linears(model), ())
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                set(summary["evaluations"]), {"baseline", "compressed", "finetuned"}
            )
            self.assertEqual(len(summary["compressed_layers"]), 14)
            for stage in ("initial_artifacts", "finetuned_artifacts"):
                files = summary[stage]
                self.assertEqual(set(files), set(snapshots[1]["paths"]))
                self.assertEqual(len(set(files.values())), 14)
                for filename in files.values():
                    self.assertEqual(load_artifact(filename).representation, "mpo")
            self.assertTrue(Path(summary["checkpoint"]).is_file())
            events = [
                json.loads(line)
                for line in (root / "experiment.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [
                    event["fields"]["stage"]
                    for event in events
                    if event["event"] == "eval_completed"
                ],
                ["baseline", "compressed", "finetuned"],
            )

    def test_resume_intermediate_and_completed_checkpoints(self) -> None:
        """中间 checkpoint 续训与连续训练一致，已完成 checkpoint 无新增 loss 也可导出。"""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, uninterrupted = self.run_experiment(root / "initial")
            for step in (1, 2):
                checkpoint = root / "initial" / "checkpoints" / f"checkpoint-{step}.pt"
                output = root / f"resume-{step}"
                _, resumed = self.run_experiment(output, checkpoint)
                for name, value in uninterrupted[2]["parameters"].items():
                    torch.testing.assert_close(
                        value, resumed[2]["parameters"][name], rtol=0, atol=0
                    )
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["train_steps"], 2)
                if step == 2:
                    self.assertIsNone(summary["final_loss"])

    def test_evaluation_failure_restores_all_replaced_layers(self) -> None:
        """联合压缩后评测失败时，仍恢复全部原始 Linear。"""

        with TemporaryDirectory() as directory:
            model, snapshots = self.run_experiment(
                Path(directory), fail_evaluation=True
            )
            self.assertEqual(len(snapshots[1]["paths"]), 14)
            self.assertEqual(list_tensor_network_linears(model), ())
            for name, value in model.named_parameters():
                torch.testing.assert_close(
                    value, snapshots[0]["parameters"][name], rtol=0, atol=0
                )
