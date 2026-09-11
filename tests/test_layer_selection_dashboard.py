"""验证标准结果驱动的动态页面、Spec 计算和独立来源。

主要内容：
- ``make_sources``：生成可变任务和模块数的确定性标准结果。
- 页面来源、动态形状、双向指标及输出安全校验。
"""

import copy
import json
from pathlib import Path

import pytest

from qcomp import MPOSpec
from qcomp.evaluation import (
    EvaluationTaskConfig,
    LMEvalConfig,
    evaluation_task_config_to_dict,
)
from qcomp.workflows.sensitivity_io import (
    model_config_digest,
    write_sensitivity_results,
)
from scripts.build_layer_selection_dashboard import (
    build_dashboard,
    build_payload,
    digest,
)


def make_sources(root: Path, count: int = 5, modules: int = 252) -> dict:
    """在 root 生成 count 任务和 modules 模块，供页面与浏览器测试复用。"""
    entries = []
    types = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    names = [
        f"model.layers.{i // 7}.{'self_attn' if i % 7 < 4 else 'mlp'}.{types[i % 7]}"
        for i in range(modules)
    ]
    for index in range(count):
        task = (
            ["boolq", "mmlu", "hellaswag", "gsm8k", "triviaqa"][index]
            if index < 5
            else f"custom{index}"
        )
        path = root / task / "sensitivity_results.json"
        model = {"model_type": "synthetic", "hidden_size": 8}
        cases = []
        for i, name in enumerate(names):
            spec = (
                MPOSpec((2, 2, 2), (2, 2, 2), (1, 2, 3, 1))
                if i % 2
                else MPOSpec((2, 4), (2, 4), (1, 1, 1))
            )
            cases.append(
                {
                    "case": name,
                    "compression_plan": {
                        "targets": [
                            {
                                "module_path": name,
                                "representation": "mpo",
                                "spec": {
                                    "out_modes": list(spec.out_modes),
                                    "in_modes": list(spec.in_modes),
                                    "ranks": list(spec.ranks),
                                },
                            }
                        ]
                    },
                    "metrics": {"acc": 0.5},
                    "degradations": {"acc": 0.25},
                    "layers": {
                        name: {
                            "dense_parameters": spec.dense_num_parameters,
                            "compressed_parameters": spec.num_parameters,
                            "compression_ratio": spec.compression_ratio,
                            "relative_error": 0.1,
                        }
                    },
                    "model_ratio": 100000
                    / (100000 - spec.dense_num_parameters + spec.num_parameters),
                    "model_compression": None,
                    "compression_seconds": 1.0,
                    "evaluation_seconds": 2.0,
                }
            )
        data = {
            "kind": "qcomp_sensitivity_results",
            "run_id": task,
            "name": task,
            "status": "completed",
            "started_at": "2026-09-01T00:00:00+00:00",
            "finished_at": "2026-09-01T01:00:00+00:00",
            "model": {
                "name_or_path": "/unavailable/model",
                "model_type": "synthetic",
                "config": model,
                "config_sha256": model_config_digest(model),
                "dense_parameters": 100000,
            },
            "execution": {
                "decomposition_provider": "native",
                "execution_provider": "native",
            },
            "evaluation_config": evaluation_task_config_to_dict(
                EvaluationTaskConfig(LMEvalConfig(task), {"acc": "higher"})
            ),
            "evaluation_task": {
                "name": task,
                "dataset": "synthetic",
                "split": "test",
                "preprocessing": "fixed",
                "requested_metrics": ["acc"],
            },
            "expected_cases": names,
            "baseline": {
                "metrics": {"acc": 0.75},
                "evaluated_examples": 2,
                "total_examples": 10,
                "seconds": 1.0,
            },
            "cases": cases,
            "provenance": {"kind": "native"},
            "report_path": None,
        }
        write_sensitivity_results(path, data)
        entries.append(
            {
                "id": task,
                "label": task,
                "default_metric": "acc",
                "path": str(path.relative_to(root)),
            }
        )
    return {"datasets": entries}


@pytest.mark.parametrize("count", [1, 2, 5, 6])
def test_dynamic_tasks_specs_and_standalone_html(tmp_path, count):
    """任意任务数和混合核数使用实际参数量，HTML 无外部依赖且安全嵌入。"""
    config = make_sources(tmp_path, count=count, modules=9)
    config["datasets"][0]["label"] = "</script><b>测试"
    payload = build_payload(config, tmp_path)
    assert len(payload["datasets"]) == count and len(payload["modules"]) == 9
    assert payload["modules"][0]["compressed_parameters"] == 20
    assert payload["modules"][1]["compressed_parameters"] == 44
    assert payload["modules"][0]["saving"] == pytest.approx(44 / 100000)
    path = tmp_path / config["datasets"][0]["path"]
    before = digest(path)
    source_config = tmp_path / "sources.json"
    source_config.write_text(json.dumps(config))
    html = build_dashboard(source_config, tmp_path / "output")
    text = html.read_text()
    assert "\\u003c/script>" in text and "/* DASHBOARD_" not in text
    assert "<script src=" not in text and 'rel="stylesheet"' not in text
    assert digest(path) == before
    with pytest.raises(FileExistsError):
        build_dashboard(source_config, html.parent)


def test_invalid_sources_and_cross_source_constraints(tmp_path):
    """拒绝未完成、重复模块、数值漂移、配置及模块集合不一致。"""
    config = make_sources(tmp_path, modules=3)
    path = tmp_path / config["datasets"][1]["path"]
    original = json.loads(path.read_text())
    variants = []
    for field, value in (
        ("status", "running"),
        ("expected_cases", ["missing"]),
        ("cases", original["cases"] * 2),
    ):
        variant = copy.deepcopy(original)
        variant[field] = value
        variants.append(variant)
    for field, value in (
        ("compression_ratio", 2),
        ("dense_parameters", 5),
        ("relative_error", float("nan")),
    ):
        variant = copy.deepcopy(original)
        variant["cases"][0]["layers"][original["cases"][0]["case"]][field] = value
        variants.append(variant)
    variant = copy.deepcopy(original)
    variant["execution"]["execution_provider"] = "other"
    variants.append(variant)
    variant = copy.deepcopy(original)
    variant["cases"] = variant["cases"][:-1]
    variant["expected_cases"] = variant["expected_cases"][:-1]
    variants.append(variant)
    variant = copy.deepcopy(original)
    variant["cases"][0]["metrics"]["acc"] = 0.6
    variants.append(variant)
    for variant in variants:
        path.write_text(json.dumps(variant))
        with pytest.raises(ValueError):
            build_payload(config, tmp_path)


def test_lower_metric_non_block_paths_and_expanding_spec(tmp_path):
    """lower 指标和非 block 模块可读取，合法增参数 Spec 保留负收益。"""
    config = make_sources(tmp_path, count=1, modules=1)
    path = tmp_path / config["datasets"][0]["path"]
    data = json.loads(path.read_text())
    case = data["cases"][0]
    old = case["case"]
    case["case"] = "projection"
    case["layers"]["projection"] = case["layers"].pop(old)
    target = case["compression_plan"]["targets"][0]
    target["module_path"] = "projection"
    spec = MPOSpec((2, 4), (2, 4), (1, 4, 1))
    target["spec"]["ranks"] = [1, 4, 1]
    case["layers"]["projection"].update(
        compressed_parameters=80, compression_ratio=spec.compression_ratio
    )
    case["model_ratio"] = 100000 / 100016
    data["expected_cases"] = ["projection"]
    data["evaluation_config"]["metric_directions"]["acc"] = "lower"
    case["degradations"]["acc"] = -0.25
    write_sensitivity_results(path, data)
    payload = build_payload(config, tmp_path)
    assert payload["modules"][0]["block"] is None
    assert payload["modules"][0]["saving"] < 0
    data["baseline"]["metrics"]["acc"] = 2
    case["degradations"]["acc"] = -1.5
    write_sensitivity_results(path, data)
    with pytest.raises(ValueError, match="比例指标"):
        build_payload(config, tmp_path)


def test_different_valid_spec_is_rejected_across_tasks(tmp_path):
    """两个独立合法且形状相同的 Spec 不能跨任务混成一个模块候选。"""
    config = make_sources(tmp_path, count=2, modules=1)
    path = tmp_path / config["datasets"][1]["path"]
    data = json.loads(path.read_text())
    case = data["cases"][0]
    target = case["compression_plan"]["targets"][0]
    target["spec"]["ranks"] = [1, 2, 1]
    spec = MPOSpec(**target["spec"])
    case["layers"][case["case"]].update(
        compressed_parameters=spec.num_parameters,
        compression_ratio=spec.compression_ratio,
    )
    case["model_ratio"] = 100000 / (
        100000 - spec.dense_num_parameters + spec.num_parameters
    )
    write_sensitivity_results(path, data)
    with pytest.raises(ValueError, match="Spec 不一致"):
        build_payload(config, tmp_path)
