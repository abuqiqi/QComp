"""读取标准敏感性结果，生成不依赖模型文件或网络的离线选层页面。

主要内容：
- ``build_payload``：校验来源、对齐模块，复用 MPO Spec 计算参数量与收益。
- ``build_dashboard``：嵌入页面资源和全部结果，保存重建来源配置。

使用说明（以下命令在项目根目录执行）：
    python scripts/build_layer_selection_dashboard.py
    python scripts/build_layer_selection_dashboard.py --config config/layer_selection.json
    python scripts/build_layer_selection_dashboard.py --config config/layer_selection.json --output artifacts/layer-selection/custom-run

命令行参数（均可省略）：
- ``--config``：页面数据来源 JSON，默认使用项目根目录下的
  ``config/layer_selection.json``；默认位置由脚本位置确定，不受工作目录影响。
  显式传入的相对路径相对于当前工作目录解析。
- ``--output``：输出目录，默认使用项目根目录下的
  ``artifacts/layer-selection/<北京时间戳>/``，时间戳格式为 ``YYYYMMDDTHHMMSS``。
  显式传入的相对路径相对于当前工作目录解析；输出目录必须尚不存在。
- ``-h`` / ``--help``：查看命令行帮助。

数据来源配置：
- ``datasets`` 是非空数组，每项填写唯一的 ``id``、显示名称 ``label``、
  ``default_metric`` 和 ``path``；数据集数量不固定，完整示例见默认配置文件。
- ``path`` 指向完整的 ``sensitivity_results.json``，允许绝对路径；相对路径以
  来源配置文件所在目录为基准。生成页面不读取模型权重、数据集、报告或旧事件日志。
- 各来源必须使用相同模型和压缩执行配置，覆盖相同模块集合；同一模块的完整
  MPO Spec 和参数统计必须一致。不同模块可以使用不同 modes、核数和 ranks。
- 页面支持方向为 ``higher`` 或 ``lower`` 的 0～1 比例指标。

输出与更新：
    输出目录包含嵌入全部数据的 ``index.html`` 和保存绝对来源路径的
    ``sources.json``。页面可直接离线打开；更换来源数据或页面代码后需重新生成。
    调整页面中的数据集勾选、指标、权重或期望矩阵数无需重新运行脚本。
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

from qcomp.workflows import compression_plan_from_dict
from qcomp.workflows.sensitivity_io import read_sensitivity_results

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("layer_selection")


def digest(path: Path) -> str:
    """计算 path 文件的 SHA-256，用于嵌入来源及导入复算。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """条件 condition 不成立时拒绝生成并说明 message。"""
    if not condition:
        raise ValueError(message)


def build_payload(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """读取 config 中相对于 base 的完整结果，构建动态任务和模块 payload。"""
    entries = config["datasets"]
    require(isinstance(entries, list) and bool(entries), "至少需要一个数据集")
    ids = [entry["id"] for entry in entries]
    require(
        all(isinstance(i, str) and i.strip() for i in ids)
        and len(ids) == len(set(ids)),
        "数据集 ID 为空或重复",
    )
    datasets, reference, expected_names, reference_cases = [], None, [], {}
    for entry in entries:
        path = (base / entry["path"]).resolve()
        source = read_sensitivity_results(path)
        cases = {case["case"]: case for case in source["cases"]}
        baseline = source["baseline"]["metrics"]
        directions = source["evaluation_config"]["metric_directions"]
        require(entry["default_metric"] in directions, f"{entry['id']}: 默认指标不存在")
        for metric in directions:
            require(
                all(
                    0 <= value <= 1
                    for value in [
                        baseline[metric],
                        *(c["metrics"][metric] for c in cases.values()),
                    ]
                ),
                f"{entry['id']}/{metric}: 页面只支持 0～1 比例指标",
            )
        if reference is None:
            reference, reference_cases = source, cases
            expected_names = list(source["expected_cases"])
        else:
            require(
                source["model"]["config_sha256"] == reference["model"]["config_sha256"]
                and source["model"]["name_or_path"]
                == reference["model"]["name_or_path"],
                f"{entry['id']}: 模型配置身份不一致",
            )
            require(
                source["execution"] == reference["execution"],
                f"{entry['id']}: 压缩执行配置不一致",
            )
            require(
                set(cases) == set(expected_names),
                f"{entry['id']}: 模块集合不一致；缺少 {sorted(set(expected_names)-set(cases))}；多出 {sorted(set(cases)-set(expected_names))}",
            )
            require(
                source["model"]["dense_parameters"]
                == reference["model"]["dense_parameters"],
                f"{entry['id']}: 模型参数量不一致",
            )
            for name in expected_names:
                a, b = cases[name], reference_cases[name]
                require(
                    a["compression_plan"] == b["compression_plan"],
                    f"{entry['id']}/{name}: Spec 不一致",
                )
                for field in (
                    "dense_parameters",
                    "compressed_parameters",
                    "compression_ratio",
                ):
                    require(
                        a["layers"][name][field] == b["layers"][name][field],
                        f"{entry['id']}/{name}: 参数统计不一致",
                    )
                require(
                    math.isclose(a["model_ratio"], b["model_ratio"], rel_tol=1e-9),
                    f"{entry['id']}/{name}: 模型压缩比不一致",
                )
        datasets.append(
            {
                "id": entry["id"],
                "label": entry.get("label", entry["id"]),
                "defaultMetric": entry["default_metric"],
                "directions": directions,
                "count": source["baseline"]["evaluated_examples"],
                "total": source["baseline"]["total_examples"],
                "baseline": {m: baseline[m] for m in directions},
                "values": {
                    m: [cases[n]["metrics"][m] for n in expected_names]
                    for m in directions
                },
                "source": {
                    "path": str(path),
                    "sha256": digest(path),
                    "evaluation_config": source["evaluation_config"],
                    "evaluation_task": source["evaluation_task"],
                    "provenance": source["provenance"],
                },
            }
        )
    modules = []
    total = reference["model"]["dense_parameters"]
    for name in expected_names:
        case = reference_cases[name]
        target = case["compression_plan"]["targets"][0]
        spec = compression_plan_from_dict(case["compression_plan"]).targets[0].spec
        match = re.fullmatch(r"(.*?)\.(\d+)\.(.+)", name)
        modules.append(
            {
                "name": name,
                "target": target,
                "block": int(match[2]) if match else None,
                "type": match[1] + "." + match[3] if match else name,
                "dense_parameters": spec.dense_num_parameters,
                "compressed_parameters": spec.num_parameters,
                "compression_ratio": spec.compression_ratio,
                "saved_parameters": spec.dense_num_parameters - spec.num_parameters,
                "saving": (
                    (spec.dense_num_parameters - spec.num_parameters) / total
                    if total
                    else 1 - 1 / case["model_ratio"]
                ),
            }
        )
    return {
        "model": {
            key: reference["model"][key]
            for key in ("name_or_path", "model_type", "config_sha256")
        },
        "saving_basis": "parameter_counts" if total else "historical_model_ratio",
        "datasets": datasets,
        "modules": modules,
    }


def build_dashboard(config_path: Path, output: Path | None = None) -> Path:
    """按 config_path 生成新 output 目录，嵌入全部资源并保存绝对来源路径。"""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    payload = build_payload(config, config_path.parent)
    output = output or PROJECT / "artifacts/layer-selection" / datetime.now(
        timezone(timedelta(hours=8))
    ).strftime("%Y%m%dT%H%M%S")
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
    """解析来源配置与新输出目录，生成离线页面并打印文件位置。"""
    parser = argparse.ArgumentParser(
        description="从 sensitivity_results.json 生成离线选层页面"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/layer_selection.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_dashboard(args.config, args.output).resolve())


if __name__ == "__main__":
    main()
