import json
import subprocess
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from helpers import TinyCausalLM, TinyTokenizer, make_module_set
from qwen3_tn.data import CausalLMCollator, TokenBlockDataset
from qwen3_tn.model import TTTarget, install_tt_modules
from qwen3_tn.reporting import AUTO_END, AUTO_START, update_report
from qwen3_tn.training import (
    TTFineTuneConfig,
    finetune_causal_lm,
    latest_training_checkpoint,
)


REPO = Path(__file__).resolve().parents[1]
COMPRESSION_CONFIG = (
    REPO / "configs/compression/qwen3_8b_compactifai_d96_blocks18_35.json"
)
EXPERIMENT_CONFIG = (
    REPO / "configs/experiments/qwen3_8b_compactifai_d96_blocks18_35_alpaca_600.json"
)
RUN_SCRIPT = REPO / "scripts/run_compactifai_alpaca_native.sh"


class CompactifAIConfigTests(unittest.TestCase):
    def test_108_targets_and_exact_parameter_totals(self):
        value = json.loads(COMPRESSION_CONFIG.read_text())
        targets = [TTTarget.from_dict(item) for item in value["targets"]]
        paths = [target.module_path for target in targets]
        self.assertEqual(len(paths), 108)
        self.assertEqual(len(set(paths)), 108)
        expected = {
            f"model.layers.{layer}.{suffix}"
            for layer in range(18, 36)
            for suffix in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            )
        }
        self.assertEqual(set(paths), expected)
        self.assertTrue(all(target.spec.ranks == (1, 96, 96, 1) for target in targets))
        self.assertEqual(
            sum(target.spec.dense_num_parameters for target in targets),
            2_566_914_048,
        )
        self.assertEqual(
            sum(target.spec.num_parameters for target in targets), 218_972_160
        )

    def test_alpaca_experiment_is_native_tt_only_plan(self):
        value = json.loads(EXPERIMENT_CONFIG.read_text())
        self.assertEqual(value["backend"], "native")
        self.assertEqual(value["resume_from"], "latest")
        self.assertNotIn("evaluation", value)
        self.assertEqual(
            value["data"],
            {
                "type": "huggingface",
                "dataset": "tatsu-lab/alpaca",
                "split": "train",
                "text_field": "text",
            },
        )
        training = value["training"]
        self.assertEqual(training["max_steps"], 600)
        self.assertEqual(training["max_length"], 512)
        self.assertEqual(training["gradient_accumulation_steps"], 4)
        self.assertEqual(training["save_steps"], 150)
        self.assertEqual(training["save_total_limit"], 2)
        self.assertEqual(training["first_step_peak_memory_limit_gib"], 80.0)
        self.assertTrue(training["gradient_checkpointing"])
        self.assertTrue(training["tt_activation_checkpointing"])

    def test_background_script_syntax(self):
        subprocess.run(["bash", "-n", str(RUN_SCRIPT)], check=True)


class LatestCheckpointTests(unittest.TestCase):
    def _fake_checkpoint(
        self,
        root: Path,
        step: int,
        config: TTFineTuneConfig,
        metadata: dict[str, str],
        *,
        complete: bool = True,
    ) -> Path:
        path = root / f"checkpoint-{step}"
        (path / "tt_modules").mkdir(parents=True)
        (path / "trainer_state.json").write_text(
            json.dumps(
                {
                    "format": "qwen3-tn-training-state-v1",
                    "state": {"global_step": step, "epoch": 0, "batch_index": step},
                    "config": asdict(config),
                    "backend_names": ["native"],
                    "metadata": metadata,
                }
            )
        )
        if complete:
            (path / "training_state.pt").touch()
            (path / "tt_modules" / "index.json").write_text("{}")
        return path

    def test_none_multiple_incomplete_and_signature_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TTFineTuneConfig(device="cpu", bf16_autocast=False)
            metadata = {"training_signature_sha256": "correct"}
            self.assertIsNone(
                latest_training_checkpoint(
                    root, expected_config=config, expected_metadata=metadata
                )
            )
            self._fake_checkpoint(root, 10, config, metadata)
            self._fake_checkpoint(
                root, 20, config, {"training_signature_sha256": "old"}
            )
            self._fake_checkpoint(root, 30, config, metadata, complete=False)
            expected = self._fake_checkpoint(root, 25, config, metadata)
            self.assertEqual(
                latest_training_checkpoint(
                    root, expected_config=config, expected_metadata=metadata
                ),
                expected,
            )

    def test_completed_step_can_be_reexported_without_an_update(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TTFineTuneConfig(
                max_steps=1,
                gradient_accumulation_steps=1,
                device="cpu",
                bf16_autocast=False,
                logging_steps=1,
                save_steps=0,
            )
            dataset = TokenBlockDataset([[1, 2, 3], [4, 5]])
            loader = DataLoader(
                dataset, batch_size=1, collate_fn=CausalLMCollator(TinyTokenizer())
            )
            first = TinyCausalLM()
            source = make_module_set(root / "source", first)
            install_tt_modules(first, source, trainable=True, core_dtype=torch.float32)
            first_metrics = finetune_causal_lm(
                first, loader, config, root / "run", model_path="tiny"
            )
            self.assertGreater(first_metrics["training_tokens_seen"], 0)
            second = TinyCausalLM()
            install_tt_modules(second, source, trainable=True, core_dtype=torch.float32)
            metrics = finetune_causal_lm(
                second,
                loader,
                config,
                root / "reexport",
                model_path="tiny",
                resume_from=root / "run" / "checkpoint-1",
            )
            self.assertEqual(metrics["global_step"], 1)
            self.assertEqual(metrics["optimizer_steps_this_run"], 0)
            self.assertEqual(metrics["training_tokens_this_run"], 0)
            self.assertEqual(
                metrics["training_tokens_seen"], first_metrics["training_tokens_seen"]
            )
            self.assertTrue((root / "reexport" / "final" / "index.json").is_file())


class WeeklyReportTests(unittest.TestCase):
    def test_five_states_preserve_manual_content_and_links(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.md"
            report.write_text(
                "人工说明：不要覆盖。\n\n"
                + AUTO_START
                + "\n旧内容\n"
                + AUTO_END
                + "\n\n人工结论：也不要覆盖。\n"
            )
            artifact = root / "artifact"
            checkpoint = artifact / "checkpoint-150"
            checkpoint.mkdir(parents=True)
            final_index = artifact / "final" / "index.json"
            final_index.parent.mkdir()
            final_index.write_text("{}")
            log = root / "run.log"
            log.write_text("traceback\nboom\n")
            run_json = root / "run.json"
            decomposition_json = root / "decomposition.json"
            verification_json = root / "verification.json"
            decomposition_json.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "dense_parameters": 2_566_914_048,
                            "tt_parameters": 218_972_160,
                            "compression_ratio": 11.7225589,
                        }
                    }
                )
            )
            verification_json.write_text(
                json.dumps({"success": True, "generated_text": "测试生成成功"})
            )
            base_state = {
                "stage": "测试",
                "artifact_root": str(artifact),
                "log_path": str(log),
                "run_json": str(run_json),
                "decomposition_json": str(decomposition_json),
                "verification_json": str(verification_json),
                "start_time": "2026-08-27T10:00:00+00:00",
                "end_time": "2026-08-27T11:00:00+00:00",
            }
            for status, expected in (
                ("pending", "未开始"),
                ("running", "运行中"),
                ("failure", "失败阶段"),
                ("success", "当前状态：成功"),
            ):
                run_json.write_text(json.dumps({"training_metrics": {}}))
                update_report(
                    report,
                    {**base_state, "status": status},
                    run_json=run_json,
                    decomposition_json=decomposition_json,
                    verification_json=verification_json,
                )
                text = report.read_text()
                self.assertIn(expected, text)
                self.assertIn("人工说明：不要覆盖。", text)
                self.assertIn("人工结论：也不要覆盖。", text)
                self.assertEqual(text.count(AUTO_START), 1)
                self.assertEqual(text.count(AUTO_END), 1)
            run_json.write_text(
                json.dumps(
                    {
                        "training_metrics": {
                            "global_step": 600,
                            "first_training_loss": 3.0,
                            "last_training_loss": 2.0,
                            "min_training_loss": 1.9,
                            "resumed_from": str(checkpoint),
                            "peak_cuda_allocated_bytes": 10 * 1024**3,
                        }
                    }
                )
            )
            log.write_text(
                "step=10 loss=3.500000 lr=1e-5\nstep=600 loss=2.200000 lr=0\n"
            )
            update_report(
                report,
                {
                    **base_state,
                    "status": "success",
                    "attempt_count": 2,
                    "last_resume_time": "2026-08-27T10:15:00+00:00",
                },
                run_json=run_json,
                decomposition_json=decomposition_json,
                verification_json=verification_json,
            )
            text = report.read_text()
            self.assertIn("当前状态：成功", text)
            self.assertIn("2026-08-27T10:15:00+00:00", text)
            self.assertIn("成功主流程耗时", text)
            self.assertNotIn("次中断", text)
            self.assertIn("测试生成成功", text)
            self.assertIn("run.json", text)
            self.assertIn("3.5000 / 2.2000 / 2.2000", text)
            self.assertIn("-1.3000", text)


if __name__ == "__main__":
    unittest.main()
