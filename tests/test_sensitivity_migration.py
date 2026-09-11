"""验证历史日志转换、合并来源和原文件保留。

主要内容：
- ``make_legacy_sources``：构造确定性历史日志。
- ``LegacySourceTests``：校验原始与合并输入。
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from qcomp.workflows.sensitivity_io import read_sensitivity_results
from scripts.migrate_sensitivity_results import (
    convert_source,
    digest,
    migrate,
    read_source,
)

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def make_legacy_sources(root: Path) -> dict:
    """在 root 构造五份完整实验，返回相对路径配置。"""
    model_dir = root / "model"
    model_dir.mkdir()
    model_config = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,
    }
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
        config = {
            "name": f"qwen3-{task}-mpo-rank-96",
            "task": task,
            "metrics": ["acc"],
            "metric_directions": {"acc": "higher"},
            "layers": 252,
            "run_id": task,
            "model_name_or_path": str(model_dir),
            "model_dtype": "torch.bfloat16",
            "decomposition_dtype": "torch.float32",
            "decomposition_provider": "tensorly",
            "execution_provider": "tensorly",
            "trust_remote_code": False,
            "num_fewshot": 0,
            "seed": 42,
            "apply_chat_template": False,
            "batch_size": 8,
            "max_length": 4096,
            "limit": 2,
            "sample_start_index": 0,
        }
        records = []
        for block in range(36):
            for index, module in enumerate(MODULES):
                name = f"model.layers.{block}.{'self_attn' if index < 4 else 'mlp'}.{module}"
                records.append(
                    {
                        "run_id": task,
                        "case": name,
                        "metrics": {"acc": 0.5},
                        "degradations": {"acc": 0.25},
                        "model_ratio": 1.001,
                        "compression_seconds": 1.0,
                        "evaluation_seconds": 2.0,
                        "layers": {
                            name: {
                                "compression_ratio": ratios[module],
                                "relative_error": 0.1,
                            }
                        },
                    }
                )
        events = [{"event": "experiment_started", "fields": config}]
        events += [{"event": "case_completed", "fields": r} for r in records]
        events.append(
            {
                "event": "experiment_completed",
                "fields": {"run_id": task, "report": "report.md"},
            }
        )
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )
        (directory / "report.md").write_text(
            "- Evaluated examples: 2\n- Total evaluation examples: 10\n\n## Baseline Metrics\n| acc | 0.75 |\n"
        )
        entries.append(
            {"id": task, "label": task, "metric": "acc", "path": f"{task}/events.jsonl"}
        )
    return {"datasets": entries}


class LegacySourceTests(unittest.TestCase):
    """历史读取器的合并与数值校验。"""

    def test_invalid_records_rejected(self) -> None:
        """未完成、重复、非有限数和基线漂移必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_legacy_sources(root)
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

    def test_merged_layer_sources_are_verified(self) -> None:
        """按层合并逐条核对来源，并拒绝伪造哈希。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_legacy_sources(root)
            source = root / "boolq/events.jsonl"
            loaded = read_source(source)
            merged = root / "merged"
            merged.mkdir()
            (merged / "report.md").write_text((source.parent / "report.md").read_text())
            document = {
                "kind": "merged_layer_sensitivity",
                "configuration": loaded["config"],
                "layers": 252,
                "evaluated_examples": 2,
                "total_examples": 10,
                "baseline_metrics": loaded["baseline"],
                "cases": list(loaded["cases"].values()),
                "sources": [{"log": str(source), "sha256": digest(source)}],
            }
            path = merged / "merged_results.json"
            path.write_text(json.dumps(document))
            self.assertEqual(len(read_source(path)["cases"]), 252)
            converted = convert_source(path, root, root / "converted", {})
            result = read_sensitivity_results(converted)
            self.assertEqual(len(result["provenance"]["converted_sources"]), 1)
            self.assertEqual(len(result["cases"]), 252)
            document["sources"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "哈希"):
                read_source(path)

    def test_sample_merge_verifies_weighted_values(self) -> None:
        """按样本合并核对不重叠范围及加权数值，不能只相信合并标签。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_legacy_sources(root)
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
                sources.append({"path": str(path), "sha256": digest(path)})
            merged = root / "merged"
            merged.mkdir()
            events = copy.deepcopy(original)
            events[0]["fields"].update(
                limit=4,
                aggregation={
                    "method": "sample_count_weighted_mean",
                    "sources": sources,
                },
            )
            for event in events[1:-1]:
                event["fields"]["metrics"]["acc"] = 0.375
            path = merged / "events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events))
            (merged / "report.md").write_text(
                "- Evaluated examples: 4\n- Total evaluation examples: 10\n\n## Baseline Metrics\n| acc | .625 |\n"
            )
            self.assertEqual(read_source(path)["baseline"], {"acc": 0.625})
            converted = convert_source(path, root, root / "converted", {})
            result = read_sensitivity_results(converted)
            self.assertEqual(result["baseline"]["metrics"], {"acc": 0.625})
            self.assertEqual(len(result["provenance"]["converted_sources"]), 2)
            events[1]["fields"]["metrics"]["acc"] = 0.5
            events[1]["fields"]["degradations"]["acc"] = 0.125
            path.write_text("\n".join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError, "加权合并"):
                read_source(path)


def test_migration_preserves_sources_and_skips_incomplete(tmp_path):
    """完整转换不改原始哈希，中断记录跳过，输出配置只引用新 JSON。"""
    config = make_legacy_sources(tmp_path)
    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps(config))
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "events.jsonl").write_text('{"event":"experiment_started","fields":{}}')
    manifest = migrate(tmp_path, tmp_path / "migrated-test", config_path)
    assert manifest["originals_unchanged"]
    assert [r["status"] for r in manifest["records"]].count("converted") == 5
    assert [r["status"] for r in manifest["records"]].count("skipped") == 1
    data = read_sensitivity_results(
        tmp_path / "migrated-test/boolq/sensitivity_results.json"
    )
    assert data["cases"][1]["compression_plan"]["targets"][0]["spec"]["ranks"] == [
        1,
        96,
        96,
        1,
    ]
    assert data["model"]["dense_parameters"] is None
    assert data["evaluation_config"]["evaluation"]["num_fewshot"] == 0
    again = migrate(
        tmp_path, tmp_path / "migrated-second", Path(manifest["dashboard_config"])
    )
    assert len(again["records"]) == 6
    assert again["dashboard_config"] is not None


def test_failed_complete_source_is_reported_independently(tmp_path):
    """完整记录缺少必要来源时标记失败，其他实验继续转换且不生成不完整页面配置。"""
    config = make_legacy_sources(tmp_path)
    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "boolq/report.md").unlink()
    manifest = migrate(tmp_path, tmp_path / "migrated-test", config_path)
    assert sum(r["status"] == "failed" for r in manifest["records"]) == 1
    assert sum(r["status"] == "converted" for r in manifest["records"]) == 4
    assert manifest["dashboard_config"] is None
    assert not (tmp_path / "migrated-test/boolq/sensitivity_results.json").exists()
