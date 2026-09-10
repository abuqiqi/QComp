"""验证离线选层的数据完整性、来源追溯和独立页面生成。

主要内容：
- ``make_sources``：构造五任务、252 个模块的小型确定性日志。
- ``DashboardSourceTests``：验证配置、指标、来源哈希和安全嵌入。
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_layer_selection_dashboard import (
    MODULES,
    build_dashboard,
    build_payload,
    digest,
    read_source,
)


def make_sources(root: Path) -> dict:
    """在 root 构造五份完整实验，返回相对路径配置。"""
    model_dir = root / "model"
    model_dir.mkdir()
    model_config = dict(
        model_type="qwen3",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=36,
    )
    (model_dir / "config.json").write_text(json.dumps(model_config))
    ratios = {
        "q_proj": 6.965986394557823,
        "o_proj": 6.965986394557823,
        "k_proj": 3.4478114478114477,
        "v_proj": 3.4478114478114477,
        "gate_proj": 20.48,
        "up_proj": 20.48,
        "down_proj": 20.48,
    }
    entries = []
    for task in ["boolq", "mmlu", "hellaswag", "gsm8k", "triviaqa"]:
        directory = root / task
        directory.mkdir()
        config = dict(
            name=f"qwen3-{task}-mpo-rank-96",
            task=task,
            metrics=["acc"],
            metric_directions={"acc": "higher"},
            layers=252,
            run_id=task,
            model_name_or_path=str(model_dir),
            model_dtype="torch.bfloat16",
            decomposition_dtype="torch.float32",
            decomposition_provider="tensorly",
            execution_provider="tensorly",
            trust_remote_code=False,
            num_fewshot=0,
            seed=42,
            apply_chat_template=False,
            batch_size=8,
            max_length=4096,
            limit=2,
            sample_start_index=0,
        )
        records = []
        for block in range(36):
            for index, module in enumerate(MODULES):
                name = f"model.layers.{block}.{'self_attn' if index < 4 else 'mlp'}.{module}"
                records.append(
                    dict(
                        run_id=task,
                        case=name,
                        metrics={"acc": 0.5},
                        degradations={"acc": 0.25},
                        model_ratio=1.001,
                        layers={
                            name: {
                                "compression_ratio": ratios[module],
                                "relative_error": 0.1,
                            }
                        },
                    )
                )
        events = [dict(event="experiment_started", fields=config)]
        events += [dict(event="case_completed", fields=r) for r in records]
        events.append(
            dict(
                event="experiment_completed",
                fields={"run_id": task, "report": "report.md"},
            )
        )
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )
        (directory / "report.md").write_text(
            "- Evaluated examples: 2\n- Total evaluation examples: 10\n\n## Baseline Metrics\n| acc | 0.75 |\n"
        )
        entries.append(
            dict(id=task, label=task, metric="acc", path=f"{task}/events.jsonl")
        )
    return {"datasets": entries}


class DashboardSourceTests(unittest.TestCase):
    """验证来源完整性以及输出不改变原始实验。"""

    def test_payload_and_standalone_html(self) -> None:
        """五份来源对齐，哈希可追溯，HTML 嵌入资源且文本安全编码。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_sources(root)
            config["datasets"][0]["label"] = "</script><b>测试"
            payload = build_payload(config, root)
            self.assertEqual(len(payload["modules"]), 252)
            self.assertEqual(payload["datasets"][0]["baseline"], {"acc": 0.75})
            self.assertAlmostEqual(payload["modules"][0]["saving"], 1 - 1 / 1.001)
            source = root / "boolq/events.jsonl"
            before = digest(source)
            self.assertEqual(payload["datasets"][0]["source"]["sha256"], before)
            path = root / "sources.json"
            path.write_text(json.dumps(config))
            destination = build_dashboard(path, root / "output")
            html = destination.read_text()
            self.assertNotIn("/* DASHBOARD_", html)
            self.assertIn("\\u003c/script>", html)
            self.assertNotIn("<script src=", html)
            self.assertNotIn('rel="stylesheet"', html)
            self.assertEqual(before, digest(source))
            saved = json.loads((destination.parent / "sources.json").read_text())
            self.assertTrue(Path(saved["datasets"][0]["path"]).is_absolute())
            with self.assertRaises(FileExistsError):
                build_dashboard(path, root / "output")

    def test_model_config_and_spec_validation(self) -> None:
        """配置推导覆盖七类矩阵，缺失配置、错误 block 和压缩比均拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_sources(root)
            payload = build_payload(config, root)
            spec = payload["modules"][1]["target"]["spec"]
            self.assertEqual(spec["out_modes"], [8, 8, 16])
            self.assertEqual(spec["ranks"], [1, 96, 96, 1])
            model_path = root / "model/config.json"
            model_path.rename(root / "moved.json")
            with self.assertRaisesRegex(ValueError, "--model-config"):
                build_payload(config, root)
            self.assertEqual(
                build_payload(config, root, root / "moved.json")["model"][
                    "config_sha256"
                ],
                payload["model"]["config_sha256"],
            )
            model = json.loads((root / "moved.json").read_text())
            for key, value in [
                ("num_hidden_layers", 35),
                ("hidden_size", 1024),
                ("attention_bias", True),
                ("model_type", "other"),
            ]:
                changed = dict(model, **{key: value})
                model_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    build_payload(config, root)
            model_path.write_text(json.dumps(model))
            for task in config["datasets"]:
                log = root / task["path"]
                events = [json.loads(line) for line in log.read_text().splitlines()]
                layer = events[1]["fields"]["case"]
                events[1]["fields"]["layers"][layer]["compression_ratio"] = 2
                log.write_text("\n".join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError, "压缩比"):
                build_payload(config, root)

    def test_invalid_records_rejected(self) -> None:
        """未完成、重复、非有限数和基线漂移必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sources(root)
            path = root / "boolq/events.jsonl"
            original = [json.loads(line) for line in path.read_text().splitlines()]
            variants = [original[:-1]]
            duplicate = copy.deepcopy(original)
            duplicate[2] = duplicate[1]
            variants.append(duplicate)
            for value in [float("nan"), 0.4]:
                changed = copy.deepcopy(original)
                changed[1]["fields"]["metrics"]["acc"] = value
                variants.append(changed)
            for events in variants:
                path.write_text("\n".join(json.dumps(e) for e in events))
                with self.assertRaises(ValueError):
                    read_source(path)

    def test_cross_task_model_and_rank_validation(self) -> None:
        """跨任务模型和 rank 变化时不能生成混合方案。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_sources(root)
            path = root / "gsm8k/events.jsonl"
            original = [json.loads(line) for line in path.read_text().splitlines()]
            for key, value in [
                ("model_name_or_path", "different"),
                ("name", "qwen3-gsm8k-mpo-rank-64"),
            ]:
                events = copy.deepcopy(original)
                events[0]["fields"][key] = value
                path.write_text("\n".join(json.dumps(e) for e in events))
                with self.assertRaises(ValueError):
                    build_payload(config, root)

    def test_merged_layer_sources_are_verified(self) -> None:
        """按层合并逐条核对来源，并拒绝伪造哈希。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sources(root)
            source = root / "boolq/events.jsonl"
            loaded = read_source(source)
            merged = root / "merged"
            merged.mkdir()
            (merged / "report.md").write_text((source.parent / "report.md").read_text())
            document = dict(
                kind="merged_layer_sensitivity",
                configuration=loaded["config"],
                layers=252,
                evaluated_examples=2,
                total_examples=10,
                baseline_metrics=loaded["baseline"],
                cases=list(loaded["cases"].values()),
                sources=[dict(log=str(source), sha256=digest(source))],
            )
            path = merged / "merged_results.json"
            path.write_text(json.dumps(document))
            self.assertEqual(len(read_source(path)["cases"]), 252)
            document["sources"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "哈希"):
                read_source(path)

    def test_sample_merge_verifies_weighted_values(self) -> None:
        """按样本合并核对不重叠范围及加权数值，不能只相信合并标签。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sources(root)
            original = [
                json.loads(s)
                for s in (root / "boolq/events.jsonl").read_text().splitlines()
            ]
            sources = []
            for index in range(2):
                directory = root / f"part{index}"
                directory.mkdir()
                events = copy.deepcopy(original)
                events[0]["fields"]["sample_start_index"] = index * 2
                if index:
                    for event in events[1:-1]:
                        event["fields"]["metrics"]["acc"] = 0.25
                path = directory / "events.jsonl"
                path.write_text("\n".join(json.dumps(e) for e in events))
                (directory / "report.md").write_text(
                    "- Evaluated examples: 2\n- Total evaluation examples: 10\n\n"
                    f"## Baseline Metrics\n| acc | {'.5' if index else '.75'} |\n"
                )
                sources.append(dict(path=str(path), sha256=digest(path)))
            merged = root / "merged"
            merged.mkdir()
            events = copy.deepcopy(original)
            events[0]["fields"].update(
                limit=4,
                aggregation=dict(method="sample_count_weighted_mean", sources=sources),
            )
            for event in events[1:-1]:
                event["fields"]["metrics"]["acc"] = 0.375
            path = merged / "events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events))
            (merged / "report.md").write_text(
                "- Evaluated examples: 4\n- Total evaluation examples: 10\n\n## Baseline Metrics\n| acc | .625 |\n"
            )
            self.assertEqual(read_source(path)["baseline"], {"acc": 0.625})
            events[1]["fields"]["metrics"]["acc"] = 0.5
            events[1]["fields"]["degradations"]["acc"] = 0.125
            path.write_text("\n".join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError, "加权合并"):
                read_source(path)


if __name__ == "__main__":
    unittest.main()
