"""统一敏感性结果的结构校验和原子读写，不加载模型或评测数据。

主要内容：
- ``validate_sensitivity_results``：核对状态、完整计划、基线和压缩统计。
- ``read_sensitivity_results``、``write_sensitivity_results``：读取或原子保存结果。
- ``model_config_digest``：对配置快照生成与文件路径无关的规范哈希。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from ..evaluation import evaluation_task_config_from_dict
from .compression_plan_io import compression_plan_from_dict


def model_config_digest(config: dict[str, Any]) -> str:
    """对 config 的规范 JSON 计算 SHA-256，不依赖原文件排版。"""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    """条件 condition 不成立时抛出包含 message 的格式错误。"""
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, minimum: float | None = None) -> None:
    """校验名为 name 的有限数 value 及可选 minimum 下限。"""
    _require(
        type(value) in (int, float) and math.isfinite(value), f"{name}: 必须为有限数"
    )
    if minimum is not None:
        _require(value >= minimum, f"{name}: 小于 {minimum}")


def validate_sensitivity_results(
    data: dict[str, Any], *, completed: bool = False
) -> None:
    """验证 data 的实验边界和数值关系；completed 为真时只接受完整实验。"""
    _require(isinstance(data, dict), "结果必须为 JSON 对象")
    _require(
        data.get("kind") == "qcomp_sensitivity_results", "不是 sensitivity results"
    )
    _require(
        data.get("status") in {"running", "completed", "failed", "interrupted"},
        "未知实验状态",
    )
    if completed:
        _require(data["status"] == "completed", "实验尚未完成")
    for key in (
        "run_id",
        "name",
        "started_at",
        "finished_at",
        "execution",
        "provenance",
        "evaluation_task",
        "model",
        "evaluation_config",
        "expected_cases",
        "baseline",
        "cases",
    ):
        _require(key in data, f"缺少 {key}")
    for field in ("name", "run_id"):
        _require(
            isinstance(data[field], str) and bool(data[field].strip()),
            f"{field}: 不能为空",
        )
    model = data["model"]
    _require(
        isinstance(model, dict) and isinstance(model.get("config"), dict),
        "模型配置必须为对象",
    )
    _require(
        model.get("model_type") == model["config"].get("model_type"),
        "模型类型与配置快照不一致",
    )
    _require(
        model["config_sha256"] == model_config_digest(model["config"]),
        "模型配置哈希不一致",
    )
    total_parameters = model["dense_parameters"]
    _require(
        total_parameters is None
        or type(total_parameters) is int
        and total_parameters > 0,
        "模型参数量无效",
    )
    task = evaluation_task_config_from_dict(data["evaluation_config"])
    directions = task.metric_directions
    expected = data["expected_cases"]
    _require(
        isinstance(expected, list)
        and bool(expected)
        and all(isinstance(n, str) and n.strip() == n and n for n in expected),
        "预期 case 名单无效",
    )
    _require(len(expected) == len(set(expected)), "预期 case 重复")
    cases = data["cases"]
    _require(
        isinstance(cases, list) and all(isinstance(c, dict) for c in cases),
        "cases 必须为对象数组",
    )
    names = [c["case"] for c in cases]
    _require(
        len(names) == len(set(names)) and set(names) <= set(expected),
        "case 重复或超出预期",
    )
    if data["status"] == "completed":
        _require(set(names) == set(expected), "完成实验的 case 不完整")
    baseline = data["baseline"]
    _require(
        baseline is not None or not cases and data["status"] != "completed", "缺少基线"
    )
    if baseline is None:
        return
    count, total = baseline["evaluated_examples"], baseline["total_examples"]
    _require(type(count) is int and count > 0, "评测题数无效")
    _require(total is None or type(total) is int and total >= count, "总题数无效")
    _require(set(baseline["metrics"]) >= set(directions), "基线缺少指标")
    metadata = data["evaluation_task"]
    _require(isinstance(metadata, dict), "基线缺少实际任务信息")
    for field in ("name", "dataset", "split", "preprocessing", "requested_metrics"):
        _require(field in metadata, f"实际任务缺少 {field}")
    _require(
        set(metadata["requested_metrics"]) >= set(directions), "实际任务缺少评测指标"
    )
    if baseline.get("seconds") is not None:
        _number(baseline["seconds"], "baseline.seconds", 0)
    for metric, value in baseline["metrics"].items():
        _number(value, f"基线 {metric}")
    for case in cases:
        name = case["case"]
        plan = compression_plan_from_dict(case["compression_plan"])
        _require(
            len(plan.targets) == 1 and plan.targets[0].module_path == name,
            f"{name}: 必须为对应模块的单目标计划",
        )
        _require(set(case["layers"]) == {name}, f"{name}: 压缩统计模块不一致")
        spec, layer = plan.targets[0].spec, case["layers"][name]
        _require(
            type(layer["dense_parameters"]) is int
            and type(layer["compressed_parameters"]) is int,
            f"{name}: 参数量必须为整数",
        )
        _require(
            layer["dense_parameters"] == spec.dense_num_parameters
            and layer["compressed_parameters"] == spec.num_parameters,
            f"{name}: Spec 参数量不一致",
        )
        _require(
            math.isclose(
                layer["compression_ratio"], spec.compression_ratio, rel_tol=1e-9
            ),
            f"{name}: Spec 压缩比不一致",
        )
        _number(layer["relative_error"], f"{name} relative_error", 0)
        _number(case["model_ratio"], f"{name} model_ratio", 0)
        _require(case["model_ratio"] > 0, "模型压缩比必须大于零")
        if total_parameters is not None:
            remaining = (
                total_parameters - spec.dense_num_parameters + spec.num_parameters
            )
            _require(
                remaining > 0 and total_parameters >= spec.dense_num_parameters,
                "模型参数量小于目标",
            )
            _require(
                math.isclose(
                    case["model_ratio"], total_parameters / remaining, rel_tol=1e-9
                ),
                f"{name}: 模型压缩比不一致",
            )
        measured = case.get("model_compression")
        if measured is not None:
            _require(total_parameters is not None, f"{name}: 模型统计缺少原始总参数量")
            _require(
                measured["dense_parameters"] == total_parameters
                and measured["compressed_parameters"] == remaining
                and measured["compressed_layers"] == 1,
                f"{name}: 模型统计参数量不一致",
            )
            _require(
                math.isclose(
                    measured["compression_ratio"], case["model_ratio"], rel_tol=1e-9
                ),
                f"{name}: 模型统计压缩比不一致",
            )
        for metric, value in case["metrics"].items():
            _number(value, f"{name} {metric}")
        _require(
            set(case["degradations"]) == set(directions), f"{name}: 退化指标不匹配"
        )
        for metric, direction in directions.items():
            value, drop = case["metrics"][metric], case["degradations"][metric]
            _number(value, f"{name} {metric}")
            _number(drop, f"{name} degradation")
            expected_drop = (baseline["metrics"][metric] - value) * (
                1 if direction == "higher" else -1
            )
            _require(
                math.isclose(drop, expected_drop, rel_tol=1e-9, abs_tol=1e-12),
                f"{name}: 指标退化量不一致",
            )
        for field in ("compression_seconds", "evaluation_seconds"):
            if case[field] is not None:
                _number(case[field], f"{name} {field}", 0)


def read_sensitivity_results(
    path: str | Path, *, completed: bool = True
) -> dict[str, Any]:
    """读取 path 并校验结果；completed 默认要求实验完整，不访问外部来源。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_sensitivity_results(data, completed=completed)
    return data


def write_sensitivity_results(path: str | Path, data: dict[str, Any]) -> None:
    """将 data 校验后原子写入 path；写入失败保留上一次结果并清理临时文件。"""
    validate_sensitivity_results(data)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
