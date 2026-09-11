"""把历史 Qwen3 敏感性实验转换为独立的标准结果，保留原文件与来源关系。

主要内容：
- ``convert_source``：核验旧日志、恢复计划和任务信息，转换合并依赖。
- ``migrate``：扫描完整实验，输出清单和对应的页面来源配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qcomp import MPOSpec
from qcomp.evaluation import (
    EvaluationTaskConfig,
    LMEvalConfig,
    evaluation_task_config_to_dict,
)
from qcomp.workflows.sensitivity_io import (
    model_config_digest,
    read_sensitivity_results,
    write_sensitivity_results,
)

if __package__:
    from .qwen3_mpo_config import qwen3_mpo_spec_dict, qwen3_projection_shapes
else:
    from qwen3_mpo_config import qwen3_mpo_spec_dict, qwen3_projection_shapes

PROJECT = Path(__file__).resolve().parents[1]
MODEL_FIELDS = (
    "model_name_or_path",
    "model_dtype",
    "decomposition_dtype",
    "decomposition_provider",
    "execution_provider",
    "trust_remote_code",
)


def require(condition: bool, message: str) -> None:
    """条件不成立时拒绝继续生成，message 指明不能验证的数据。"""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """返回 path 的 SHA-256，用于记录来源和验证合并输入。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(record: dict[str, Any], parent: Path) -> Path:
    """解析合并记录路径，兼容现有 evaluations 到 sensitivity 的目录迁移。"""
    path = Path(record.get("path", record.get("log", "")))
    if not path.is_file():
        path = parent.parent / path.parent.name / path.name
    require(path.is_file(), f"合并来源不存在：{path}")
    require(digest(path) == record["sha256"], f"合并来源哈希不一致：{path}")
    return path


def read_source(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """读取 path 的完整实验，验证合并输入、样本数、指标和模块记录。

    参数：
        path: events.jsonl 或 merged_results.json。
        seen: 已访问路径，防止合并来源循环。
    返回：
        统一的配置、模块映射、基线、样本数和来源记录。
    异常：
        ValueError: 实验未完成、配置缺失或合并内容无法验证。
    """
    path = path.resolve()
    require(path not in seen, f"合并来源循环：{path}")
    report_path = path.parent / "report.md"
    report = report_path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        events = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        starts = [e["fields"] for e in events if e["event"] == "experiment_started"]
        ends = [e["fields"] for e in events if e["event"] == "experiment_completed"]
        require(len(starts) == len(ends) == 1, f"需要一次完整实验：{path}")
        config = starts[0]
        records = [e["fields"] for e in events if e["event"] == "case_completed"]
        require(
            all(r["run_id"] == config["run_id"] for r in records + ends),
            f"混合 run_id：{path}",
        )
        aggregation = config.get("aggregation")
        declared_baseline = config.get("baseline_metrics")
    else:
        data = json.loads(path.read_text())
        require(
            data.get("kind") == "merged_layer_sensitivity", f"不支持的合并格式：{path}"
        )
        config, records = data["configuration"], data["cases"]
        aggregation = {"method": "layer_concatenation", "sources": data["sources"]}
        declared_baseline = data["baseline_metrics"]
        require(data["layers"] == len(records), "合并模块数不一致")
    require(
        len(records) == config.get("layers", len(records)) and bool(records),
        f"模块数不完整：{path}",
    )
    for field in MODEL_FIELDS:
        require(field in config, f"配置缺少 {field}：{path}")
    rank = re.fullmatch(r"qwen3-[\w-]+-mpo-rank-(\d+)", config["name"])
    require(rank is not None, f"无法验证 rank：{path}")
    count_match = re.search(r"^- Evaluated examples: (\d+)$", report, re.MULTILINE)
    total_match = re.search(
        r"^- Total evaluation examples: (\d+)$", report, re.MULTILINE
    )
    require(
        count_match is not None and total_match is not None, f"报告缺少样本数：{path}"
    )
    count, total = int(count_match[1]), int(total_match[1])
    require(0 < count <= total, "样本数越界")
    offset = config.get("sample_start_index", 0)
    limit = config.get("limit")
    require(isinstance(offset, int) and 0 <= offset < total, "样本起点无效")
    if limit is None:
        require(count == total - offset, "全量报告样本数与配置不一致")
    elif isinstance(limit, int):
        require(count == min(limit, total - offset), "报告样本数与 limit 不一致")
    if path.suffix != ".jsonl":
        require(
            data["evaluated_examples"] == count and data["total_examples"] == total,
            "合并元数据样本数不一致",
        )
    metrics = config["metrics"]
    require(bool(metrics) and len(metrics) == len(set(metrics)), "指标为空或重复")
    require(
        all(config["metric_directions"][m] == "higher" for m in metrics),
        "页面只支持越高越好的比例指标",
    )
    cases, baseline = {}, {}
    for record in records:
        name = record["case"]
        require(
            name not in cases and list(record["layers"]) == [name],
            f"重复或非单模块实验：{name}",
        )
        require(
            math.isfinite(record["model_ratio"]) and record["model_ratio"] > 0,
            "模型压缩比无效",
        )
        layer = record["layers"][name]
        require(
            math.isfinite(layer["compression_ratio"])
            and layer["compression_ratio"] > 0,
            "模块压缩比无效",
        )
        require(
            math.isfinite(layer["relative_error"]) and layer["relative_error"] >= 0,
            "模块误差无效",
        )
        for m in metrics:
            value, drop = record["metrics"][m], record["degradations"][m]
            base = value + drop
            require(
                all(math.isfinite(v) for v in (value, drop, base)), "指标含非有限数"
            )
            require(
                -1e-12 <= value <= 1 + 1e-12 and -1e-12 <= base <= 1 + 1e-12,
                "指标不是比例得分",
            )
            require(
                abs(base - baseline.setdefault(m, base)) < 1e-9, "同一实验的基线不一致"
            )
        cases[name] = record
    section = re.search(
        r"^## Baseline Metrics\s*\n(.*?)(?=^## |\Z)", report, re.MULTILINE | re.DOTALL
    )
    require(section is not None, "报告缺少基线表")
    for m in metrics:
        cell = re.search(
            rf"^\|\s*{re.escape(m)}\s*\|\s*([^|]+)\|\s*$", section[1], re.MULTILINE
        )
        require(
            cell is not None and abs(float(cell[1]) - baseline[m]) < 1e-6,
            f"报告基线不一致：{m}",
        )
        if declared_baseline is not None:
            require(abs(declared_baseline[m] - baseline[m]) < 1e-9, "合并基线不一致")
    provenance = {
        "path": str(path),
        "sha256": digest(path),
        "report_sha256": digest(report_path),
        "configuration": config,
        "aggregation": aggregation,
    }
    if aggregation:
        inputs = [
            read_source(source_path(s, path.parent), seen | {path})
            for s in aggregation["sources"]
        ]
        require(bool(inputs), "合并来源为空")
        for item in inputs:
            require(item["total"] == total, "合并来源数据集总量不同")
            require(item["rank"] == int(rank[1]), "合并 rank 不一致")
            for key in (
                *MODEL_FIELDS,
                "task",
                "metrics",
                "num_fewshot",
                "seed",
                "apply_chat_template",
                "batch_size",
                "max_length",
            ):
                require(item["config"][key] == config[key], f"合并配置不一致：{key}")
        if aggregation["method"] == "layer_concatenation":
            union = {}
            for item in inputs:
                require(item["count"] == count, "按层合并样本数不一致")
                require(
                    item["config"].get("sample_start_index", 0)
                    == config.get("sample_start_index", 0),
                    "按层合并样本起点不同",
                )
                require(not union.keys() & item["cases"].keys(), "按层合并存在重复模块")
                union.update(item["cases"])
            require(union == cases, "按层合并内容与来源不一致")
        else:
            require(
                aggregation["method"] == "sample_count_weighted_mean", "未知合并方式"
            )
            offset = config.get("sample_start_index", 0)
            for item in inputs:
                require(
                    item["config"].get("sample_start_index", 0) == offset,
                    "样本分段重叠或缺失",
                )
                offset += item["count"]
                require(item["cases"].keys() == cases.keys(), "样本分段模块不一致")
            require(sum(i["count"] for i in inputs) == count, "合并样本数不一致")
            for name, record in cases.items():
                for item in inputs:
                    require(
                        item["cases"][name]["layers"] == record["layers"]
                        and item["cases"][name]["model_ratio"] == record["model_ratio"],
                        "合并压缩配置不一致",
                    )
                for m in metrics:
                    expected = (
                        sum(i["cases"][name]["metrics"][m] * i["count"] for i in inputs)
                        / count
                    )
                    expected_base = (
                        sum(i["baseline"][m] * i["count"] for i in inputs) / count
                    )
                    require(
                        abs(expected - record["metrics"][m]) < 1e-9
                        and abs(expected_base - baseline[m]) < 1e-9,
                        "加权合并数值不一致",
                    )
        provenance["verified_sources"] = [i["provenance"] for i in inputs]
    return {
        "config": config,
        "rank": int(rank[1]),
        "cases": cases,
        "baseline": baseline,
        "count": count,
        "total": total,
        "provenance": provenance,
    }


def convert_source(
    path: Path, root: Path, output: Path, converted: dict[Path, Path]
) -> Path:
    """验证 path 并转换到 output 中相对 root 的目录；converted 缓存已转换依赖。"""
    path = path.resolve()
    if path in converted:
        return converted[path]
    source = read_source(path)
    cfg = source["config"]
    config_path = Path(cfg["model_name_or_path"]) / "config.json"
    model_data = json.loads(config_path.read_text())
    shapes = qwen3_projection_shapes(model_data)
    evaluation = {
        key: cfg[key]
        for key in (
            "task",
            "num_fewshot",
            "batch_size",
            "max_length",
            "limit",
            "apply_chat_template",
        )
    }
    evaluation.update(
        evaluation_seed=cfg["seed"], sample_start_index=cfg.get("sample_start_index", 0)
    )
    task_config = evaluation_task_config_to_dict(
        EvaluationTaskConfig(LMEvalConfig(**evaluation), cfg["metric_directions"])
    )
    report = (path.parent / "report.md").read_text()
    metadata = {}
    for field, heading in (
        ("name", "Task"),
        ("dataset", "Dataset"),
        ("split", "Split"),
        ("preprocessing", "Preprocessing"),
    ):
        match = re.search(rf"^- {heading}: `([^`]+)`", report, re.MULTILINE)
        metadata[field] = match[1] if match else None
    metadata["requested_metrics"] = list(cfg["metric_directions"])
    cases = []
    for name, original in source["cases"].items():
        match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)", name)
        if match is None or int(match[1]) >= model_data["num_hidden_layers"]:
            raise ValueError(f"无法恢复模块形状：{name}")
        spec_dict = qwen3_mpo_spec_dict(*shapes[match[3]], source["rank"])
        spec = MPOSpec(**spec_dict)
        layer = {
            **original["layers"][name],
            "dense_parameters": spec.dense_num_parameters,
            "compressed_parameters": spec.num_parameters,
        }
        cases.append(
            {
                **original,
                "compression_plan": {
                    "targets": [
                        {
                            "module_path": name,
                            "representation": "mpo",
                            "spec": spec_dict,
                        }
                    ]
                },
                "layers": {name: layer},
                "model_compression": None,
            }
        )
    provenance = {
        "kind": "legacy_conversion",
        "original": source["provenance"],
        "converted_at": datetime.now(UTC).isoformat(),
        "model_config_path": str(config_path),
        "model_config_file_sha256": digest(config_path),
        "spec_recovery": {
            "rule": "Qwen3 三核 modes；两条内部 bond 使用实验名称中的 rank",
            "reference_commit": "615a10a",
            "rule_sha256": digest(Path(__file__).with_name("qwen3_mpo_config.py")),
        },
        "unknown_fields": ["model.dense_parameters", "cases.model_compression"],
        "assumptions": (
            []
            if "sample_start_index" in cfg
            else ["历史脚本未提供题目偏移参数，sample_start_index=0"]
        ),
        "converted_sources": [],
    }
    provenance["unknown_fields"].extend(
        f"evaluation_task.{k}" for k, v in metadata.items() if v is None
    )
    if cfg.get("aggregation") or source["provenance"].get("aggregation"):
        aggregation = source["provenance"]["aggregation"]
        for item in aggregation["sources"]:
            original_path = source_path(item, path.parent)
            new_path = convert_source(original_path, root, output, converted)
            provenance["converted_sources"].append(
                {
                    "path": str(new_path.resolve()),
                    "sha256": digest(new_path),
                    "original_path": str(original_path.resolve()),
                }
            )
    started_at = finished_at = None
    if path.suffix == ".jsonl":
        events = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        started_at = next(
            e.get("timestamp") for e in events if e["event"] == "experiment_started"
        )
        finished_at = next(
            e.get("timestamp") for e in events if e["event"] == "experiment_completed"
        )
    if started_at is None:
        provenance["unknown_fields"].append("started_at")
    if finished_at is None:
        provenance["unknown_fields"].append("finished_at")
    timing = re.search(
        r"^- Baseline evaluation seconds: ([\d.eE+-]+)$", report, re.MULTILINE
    )
    document = {
        "kind": "qcomp_sensitivity_results",
        "run_id": cfg.get("run_id", "merged-" + digest(path)[:16]),
        "name": cfg["name"],
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "expected_cases": list(source["cases"]),
        "model": {
            "name_or_path": cfg["model_name_or_path"],
            "model_type": model_data["model_type"],
            "config": model_data,
            "config_sha256": model_config_digest(model_data),
            "dense_parameters": None,
        },
        "execution": {
            key: cfg[key] for key in MODEL_FIELDS if key != "model_name_or_path"
        },
        "evaluation_config": task_config,
        "evaluation_task": metadata,
        "baseline": {
            "metrics": source["baseline"],
            "evaluated_examples": source["count"],
            "total_examples": source["total"],
            "evaluated_tokens": None,
            "seconds": float(timing[1]) if timing else None,
        },
        "cases": cases,
        "provenance": provenance,
        "report_path": None,
    }
    destination = output / path.relative_to(root).parent / "sensitivity_results.json"
    write_sensitivity_results(destination, document)
    converted[path] = destination
    return destination


def migrate(root: Path, output: Path, config_path: Path) -> dict[str, Any]:
    """扫描 root，把记录转换到新 output 并按 config_path 生成页面配置与清单。"""
    root, output, config_path = root.resolve(), output.resolve(), config_path.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and (p.suffix == ".jsonl" or p.name == "merged_results.json")
        and not any(part.startswith("migrated-") for part in p.relative_to(root).parts)
        and output not in p.parents
    )
    before = {
        str(p): digest(p)
        for p in root.rglob("*")
        if p.is_file()
        and output not in p.parents
        and not any(part.startswith("migrated-") for part in p.relative_to(root).parts)
    }
    converted, records = {}, []
    for path in paths:
        try:
            if path.suffix == ".jsonl":
                events = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                if not any(e.get("event") == "experiment_completed" for e in events):
                    records.append(
                        {"path": str(path), "status": "skipped", "reason": "实验未完成"}
                    )
                    continue
            destination = convert_source(path, root, output, converted)
            records.append(
                {
                    "path": str(path),
                    "status": "converted",
                    "output": str(destination),
                    "sha256": digest(destination),
                }
            )
        except Exception as error:  # noqa: BLE001 - 每份结果独立转换并在清单报告失败
            records.append(
                {"path": str(path), "status": "failed", "reason": str(error)}
            )
    config = json.loads(config_path.read_text())
    entries = []
    for entry in config["datasets"]:
        source = (config_path.parent / entry["path"]).resolve()
        if source.name == "sensitivity_results.json":
            previous = read_sensitivity_results(source)
            original_path = previous["provenance"].get("original", {}).get("path")
            if original_path:
                source = Path(original_path).resolve()
        if source not in converted:
            break
        entries.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "path": str(converted[source].relative_to(output)),
                "default_metric": entry.get("default_metric", entry.get("metric")),
            }
        )
    if len(entries) == len(config["datasets"]):
        (output / "layer_selection.json").write_text(
            json.dumps({"datasets": entries}, ensure_ascii=False, indent=2) + "\n"
        )
    unchanged = all(
        Path(p).is_file() and digest(Path(p)) == sha for p, sha in before.items()
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "records": records,
        "original_sha256": before,
        "originals_unchanged": unchanged,
        "dashboard_config": (
            str(output / "layer_selection.json")
            if len(entries) == len(config["datasets"])
            else None
        ),
    }
    (output / "migration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    """解析输入与新输出目录，执行迁移并在验证失败时返回非零状态。"""
    parser = argparse.ArgumentParser(
        description="转换历史 sensitivity 结果，不修改原文件"
    )
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts/sensitivity")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/layer_selection.json"
    )
    args = parser.parse_args()
    output = args.output or args.root / (
        "migrated-"
        + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    )
    result = migrate(args.root, output, args.config)
    print(output.resolve())
    for record in result["records"]:
        print(record["status"], record["path"], record.get("reason", ""))
    if not result["originals_unchanged"] or any(
        r["status"] == "failed" for r in result["records"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
