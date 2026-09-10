"""验证计划 JSON 的往返及加载时的模型检查。

主要内容：
- 混合 rank 计划保持顺序，加载不读取来源或修改模型。
- 非法结构、未知表示、目标类型及矩阵维度必须明确报错。
"""

import copy
import json

import pytest
import torch
from torch import nn

from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    compression_plan_from_dict,
    compression_plan_to_dict,
    load_compression_plan,
)


@pytest.fixture
def plan():
    """构造两个小矩阵使用不同 rank 的有序计划。"""
    return CompressionPlan(
        tuple(
            CompressionTarget(name, "mpo", MPOSpec((2, 2), (2, 2), (1, rank, 1)))
            for name, rank in [("second", 2), ("first", 1)]
        )
    )


def test_roundtrip_and_automatic_validation(tmp_path, plan):
    """完整文件加载自动验证模型，来源缺失不影响执行，权重与模块身份不变。"""
    data = compression_plan_to_dict(plan)
    assert compression_plan_from_dict(json.loads(json.dumps(data))) == plan
    model = nn.ModuleDict(
        {name: nn.Linear(4, 4, bias=False) for name in ("first", "second")}
    )
    with torch.no_grad():
        for layer in model.values():
            layer.weight.copy_(torch.arange(16).reshape(4, 4))
    layers = dict(model.items())
    weights = {name: layer.weight.clone() for name, layer in model.items()}
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            dict(
                kind="qcomp_compression_selection",
                compression_plan=data,
                selection={"sources": [{"path": "/does/not/exist"}]},
            )
        )
    )
    assert load_compression_plan(path, model=model) == plan
    for name, layer in model.items():
        assert layer is layers[name]
        assert torch.equal(weights[name], layer.weight)


def test_invalid_structures(plan):
    """拒绝未知执行字段、重复目标、非法数值、非法 rank 和表示。"""
    original = compression_plan_to_dict(plan)
    variants = [
        {"targets": []},
        {"targets": original["targets"], "rank": 2},
        {"targets": original["targets"] * 2},
    ]
    for location, field, value in [
        ("target", "representation", "tucker"),
        ("target", "unknown", 1),
        ("target", "module_path", " "),
        ("spec", "ranks", [1, 5, 1]),
        ("spec", "ranks", [1, True, 1]),
        ("spec", "in_modes", [2, 2.0]),
        ("spec", "out_modes", []),
        ("spec", "ranks", [2, 2, 1]),
        ("spec", "ranks", [1, 2]),
        ("spec", "in_modes", [4]),
        ("spec", "unknown", 1),
    ]:
        data = copy.deepcopy(original)
        obj = data["targets"][0] if location == "target" else data["targets"][0]["spec"]
        obj[field] = value
        variants.append(data)
    for data in variants:
        with pytest.raises(ValueError):
            compression_plan_from_dict(data)


@pytest.mark.parametrize("problem", ["missing", "bias", "type", "shape"])
def test_model_mismatch(tmp_path, plan, problem):
    """所有模型不匹配由 load 内部拒绝，并指出目标路径。"""
    model = nn.ModuleDict({"first": nn.Linear(4, 4, bias=False)})
    if problem != "missing":
        model["second"] = (
            nn.ReLU()
            if problem == "type"
            else nn.Linear(3 if problem == "shape" else 4, 4, bias=problem == "bias")
        )
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            dict(
                kind="qcomp_compression_selection",
                compression_plan=compression_plan_to_dict(plan),
            )
        )
    )
    with pytest.raises(ValueError, match="second"):
        load_compression_plan(path, model=model)


def test_old_format_rejected(tmp_path):
    """旧快照必须重新导出，不自动猜测压缩结构。"""
    path = tmp_path / "old.json"
    path.write_text('{"rank": 96, "selectedModules": []}')
    with pytest.raises(ValueError, match="最新版"):
        load_compression_plan(path, model=nn.Module())
