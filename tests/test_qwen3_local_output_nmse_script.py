"""验证 Qwen3 local-output NMSE 脚本的配置、目标和候选计划。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qcomp.evaluation import (
    CompressionMetrics,
    EvaluationResult,
    EvaluationTask,
    ModelCompressionMetrics,
)
from qcomp.workflows import CompressionPlanEvaluation, TimedEvaluation
from scripts import run_qwen3_local_output_nmse as script


class Attention(nn.Module):
    """建立一个 Qwen3 block 的四个 attention 投影。"""

    def __init__(self) -> None:
        """使用 meta 参数避免测试分配大矩阵。"""
        super().__init__()
        self.q_proj = nn.Linear(4096, 4096, bias=False, device="meta")
        self.k_proj = nn.Linear(4096, 1024, bias=False, device="meta")
        self.v_proj = nn.Linear(4096, 1024, bias=False, device="meta")
        self.o_proj = nn.Linear(4096, 4096, bias=False, device="meta")


class MLP(nn.Module):
    """建立一个 Qwen3 block 的三个 MLP 投影。"""

    def __init__(self) -> None:
        """使用 Qwen3-8B 的 hidden/intermediate 形状。"""
        super().__init__()
        self.gate_proj = nn.Linear(4096, 12288, bias=False, device="meta")
        self.up_proj = nn.Linear(4096, 12288, bias=False, device="meta")
        self.down_proj = nn.Linear(12288, 4096, bias=False, device="meta")


class Block(nn.Module):
    """组合 attention 与 MLP 投影。"""

    def __init__(self) -> None:
        """创建完整七投影 block。"""
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()


class TinyQwenStructure(nn.Module):
    """提供与 Qwen3 模块路径一致的两层静态结构。"""

    def __init__(self) -> None:
        """建立 ``model.layers`` 层级。"""
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])


class CacheProbeModel(nn.Module):
    """提供自动缓存测试所需的单个投影和可计数前向。"""

    def __init__(self) -> None:
        """建立 embedding 和无 bias 投影。"""
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.projection = nn.Linear(2, 2, bias=False)
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        """执行目标投影并统计 dense reference 调用。"""
        del attention_mask, use_cache
        self.forward_calls += 1
        return self.projection(self.embedding(input_ids))


def test_select_targets_and_build_single_target_plans() -> None:
    """block 范围返回七个投影，每个保留率产生独立单目标计划。"""
    model = TinyQwenStructure()
    targets = script.select_qwen3_targets(model, start_block=1, max_blocks=1)
    assert len(targets) == 7
    assert targets[0] == "model.layers.1.self_attn.q_proj"
    assert targets[-1] == "model.layers.1.mlp.down_proj"
    plans = script.build_local_output_plans(model, targets, (0.8, 0.6, 0.4))
    assert len(plans) == 21
    assert all(len(plan.targets) == 1 for plan in plans.values())
    q_ranks = [
        plan.targets[0].spec.ranks[1]
        for plan in plans.values()
        if plan.targets[0].module_path.endswith("q_proj")
    ]
    assert q_ranks == [228, 197, 161]
    k_ranks = [
        plan.targets[0].spec.ranks[1]
        for plan in plans.values()
        if plan.targets[0].module_path.endswith("k_proj")
    ]
    assert k_ranks == [225, 194, 158]
    all_targets = script.select_qwen3_targets(model, start_block=0, max_blocks=2)
    groups = script.group_qwen3_targets_by_block(all_targets)
    assert len(groups) == 2
    assert all(len(group) == 7 for group in groups)
    assert groups[0][0].startswith("model.layers.0.")
    assert groups[1][0].startswith("model.layers.1.")


def test_heatmaps_group_sources_and_ratios_with_shared_scale(tmp_path, monkeypatch):
    """绘图按来源与保留率分组，共享色标并保留失败和超界信息。"""
    from matplotlib.figure import Figure
    import numpy as np

    records = []
    for ratio in (0.8, 0.4):
        records.append({
            "block": 0, "projection": "q_proj", "status": "ok",
            "requested_retention_ratio": ratio, "actual_retention_ratio": ratio,
            "output_nmse": 0.2, "valid_tokens": 8,
            "sources": {"sample": {"output_nmse": 1.3, "valid_tokens": 8}},
        })
    records.append({
        "block": 0, "projection": "k_proj", "status": "failed",
        "requested_retention_ratio": 0.4,
    })
    document = {
        "experiment": {"retention_ratios": [0.8, 0.4],
                       "calibration": {"datasets": [{"name": "sample"}]}},
        "dense": [{"block": 0}], "cases": records,
    }
    path = tmp_path / "local_output_nmse.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    snapshots = []

    def save(figure, destination, **kwargs):
        """捕获真实 Figure 的矩阵、色标和文本，避免测试写出大型图片。"""
        axis = figure.axes[0]
        snapshots.append((axis.images[0].get_array().copy(),
                          axis.images[0].get_clim(),
                          [text.get_text() for text in axis.texts]))

    monkeypatch.setattr(Figure, "savefig", save)
    images = script.plot_results(path)
    assert len(images) == 4
    assert "overall-rho-0.8" in images[0].name
    assert "source-sample-rho-0.8" in images[2].name
    assert all(snapshot[1] == (0, 1.3) for snapshot in snapshots)
    assert snapshots[0][0][0, 0] == pytest.approx(0.2)
    assert snapshots[2][0][0, 0] == pytest.approx(1.3)
    assert np.ma.is_masked(snapshots[0][0][1, 0])
    assert "x" in snapshots[1][2]
    script.plot_results(path, vmax=1)
    assert "1.3" in snapshots[6][2]
    report = (tmp_path / "report.md").read_text()
    assert report.count("<!-- qcomp-local-output-nmse-heatmaps -->") == 1
    assert report.index("### 总体") < report.index("### sample")
    records.append(dict(records[0]))
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        script.plot_results(path)


def test_plot_only_does_not_load_model_or_config(monkeypatch, tmp_path):
    """补图分支直接使用指定 JSON，不读取校准配置或加载模型。"""
    calls = []
    monkeypatch.setattr(script, "plot_results", lambda *args: calls.append(args))
    path = tmp_path / "results.json"
    script.main(["--plot-only", str(path), "--config", "missing.json"])
    assert calls == [(path, None)]
    with pytest.raises(SystemExit):
        script.parse_args(["--heatmap-max", "nan"])
