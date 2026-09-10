"""读取已完成的 Qwen3 敏感性结果并生成可离线打开的交互选层页面。

主要内容：
- ``read_source``：验证日志、合并来源、报告样本数和逐层基线。
- ``build_payload``：对齐五个数据集，整理浏览器使用的指标和来源。
- ``build_dashboard``：嵌入原生页面资源与数据，保存来源配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("layer_selection")
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
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
    count_match = re.search(r"^- Evaluated examples: (\d+)$", report, re.M)
    total_match = re.search(r"^- Total evaluation examples: (\d+)$", report, re.M)
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
        r"^## Baseline Metrics\s*\n(.*?)(?=^## |\Z)", report, re.M | re.S
    )
    require(section is not None, "报告缺少基线表")
    for m in metrics:
        cell = re.search(
            rf"^\|\s*{re.escape(m)}\s*\|\s*([^|]+)\|\s*$", section[1], re.M
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
    return dict(
        config=config,
        rank=int(rank[1]),
        cases=cases,
        baseline=baseline,
        count=count,
        total=total,
        provenance=provenance,
    )


def build_payload(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """按 config 的五份来源构建页面数据；相对路径基于 base 解析。"""
    entries = config["datasets"]
    require(len(entries) == 5, "配置必须包含五个数据集")
    require(
        {e["id"] for e in entries}
        == {"boolq", "mmlu", "hellaswag", "gsm8k", "triviaqa"},
        "数据集缺失或重复",
    )
    expected_names = [
        f"model.layers.{block}.{'self_attn' if index < 4 else 'mlp'}.{module}"
        for block in range(36)
        for index, module in enumerate(MODULES)
    ]
    datasets, reference = [], None
    for entry in entries:
        source = read_source(base / entry["path"])
        require(source["config"]["task"] == entry["id"], "来源 task 不匹配")
        require(source["rank"] == 96, "本页要求 rank 96")
        require(
            set(source["cases"]) == set(expected_names), "来源未覆盖全部 252 个 Linear"
        )
        require(entry["metric"] in source["baseline"], "默认指标不存在")
        if reference:
            for key in MODEL_FIELDS:
                require(
                    source["config"][key] == reference["config"][key],
                    f"跨任务配置不一致：{key}",
                )
            for name in expected_names:
                a, b = source["cases"][name], reference["cases"][name]
                require(
                    a["layers"] == b["layers"] and a["model_ratio"] == b["model_ratio"],
                    f"跨任务压缩信息不一致：{name}",
                )
        else:
            reference = source
        datasets.append(
            dict(
                id=entry["id"],
                label=entry["label"],
                defaultMetric=entry["metric"],
                count=source["count"],
                total=source["total"],
                baseline=source["baseline"],
                values={
                    m: [source["cases"][name]["metrics"][m] for name in expected_names]
                    for m in source["baseline"]
                },
                source=source["provenance"],
            )
        )
    return {
        "version": 1,
        "rank": 96,
        "datasets": datasets,
        "modules": [
            {
                "name": name,
                "block": i // 7,
                "type": MODULES[i % 7],
                "saving": 1 - 1 / reference["cases"][name]["model_ratio"],
            }
            for i, name in enumerate(expected_names)
        ],
    }


def build_dashboard(config_path: Path, output: Path | None = None) -> Path:
    """验证 config_path 并向新的 output 目录写入独立页面及可复用来源配置。"""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    payload = build_payload(config, config_path.parent)
    if output is None:
        output = (
            PROJECT
            / "artifacts/layer-selection"
            / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
        )
    html = (ASSETS / "dashboard.html").read_text()
    html = html.replace("/* DASHBOARD_CSS */", (ASSETS / "dashboard.css").read_text())
    html = html.replace("/* DASHBOARD_JS */", (ASSETS / "dashboard.js").read_text())
    html = html.replace(
        "/* DASHBOARD_DATA */",
        json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
            "<", "\\u003c"
        ),
    )
    output.mkdir(parents=True, exist_ok=False)
    for entry in config["datasets"]:
        entry["path"] = str((config_path.parent / entry["path"]).resolve())
    (output / "sources.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    return output / "index.html"


def main() -> None:
    """解析来源配置和输出目录，生成页面并打印路径。"""
    parser = argparse.ArgumentParser(description="生成离线敏感性选层页面，不加载模型。")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/layer_selection.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_dashboard(args.config, args.output).resolve())


if __name__ == "__main__":
    main()
