"""验证压缩与评测共享配置的边界、后端构造和种子兼容。

使用小型计划和 CLI 解析检查配置，不加载语言模型。
主要内容：
- 共享配置校验、表示去重和随机状态不变。
- 两个实验入口的新旧种子名称及冲突拒绝。
"""

import pytest
import torch

from qcomp import (
    CompressionExecutionConfig,
    CompressionPlan,
    CompressionTarget,
    EvaluationTaskConfig,
    MPOSpec,
)
from qcomp.evaluation import LMEvalConfig
from scripts import run_compression_plan, run_qwen3_sensitivity


def test_task_config_normalizes_and_copies():
    """规范化关注指标且不保留调用者字典引用，不保存重复指标列表。"""
    directions = {" ACC ": "higher", "loss": "lower"}
    config = EvaluationTaskConfig(LMEvalConfig(task="boolq"), directions)
    directions.clear()
    assert config.metric_directions == {"acc": "higher", "loss": "lower"}
    assert config.evaluation.evaluation_seed == 42
    with pytest.raises(TypeError):
        LMEvalConfig(task="boolq", seed=42)


@pytest.mark.parametrize(
    "directions",
    [{}, {"": "higher"}, {"acc": "up"}, {"acc": "higher", " ACC ": "lower"}],
)
def test_task_config_rejects_invalid_metrics(directions):
    """空指标、非法方向与规范化后重名必须明确拒绝。"""
    with pytest.raises(ValueError):
        EvaluationTaskConfig(LMEvalConfig(task="x"), directions)


def test_backend_config_uses_plan_representations_without_rng_changes():
    """后端按计划表示去重构建，配置过程不消耗随机状态。"""
    spec = MPOSpec.full_rank((2,), (2,))
    plan = CompressionPlan(
        (CompressionTarget("a", "mpo", spec), CompressionTarget("b", "mpo", spec))
    )
    config = CompressionExecutionConfig("native", "native")
    before = torch.get_rng_state().clone()
    decomposition, execution = config.build_backends({"first": plan, "second": plan})
    assert list(decomposition) == list(execution) == ["mpo"]
    assert (
        decomposition["mpo"].representation == execution["mpo"].representation == "mpo"
    )
    assert config.decomposition_dtype == torch.float32
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError):
        CompressionExecutionConfig(decomposition_dtype=torch.int32)
    with pytest.raises(ValueError):
        CompressionExecutionConfig(decomposition_provider=" ")
    with pytest.raises((ValueError, KeyError)):
        CompressionExecutionConfig(decomposition_provider="missing").build_backends(
            {"plan": plan}
        )


@pytest.mark.parametrize(
    "script, option, attribute, required",
    [
        (run_qwen3_sensitivity, "--evaluation-seed", "evaluation_seed", []),
        (
            run_compression_plan,
            "--decomposition-seed",
            "decomposition_seed",
            ["--selection-json", "plan.json"],
        ),
    ],
)
def test_seed_cli_compatibility(script, option, attribute, required):
    """新旧参数分别可用，默认相同，同时出现即使数值相同也拒绝。"""
    assert getattr(script.parse_args(required), attribute) == 42
    for flag in (option, "--seed"):
        assert getattr(script.parse_args(required + [flag, "17"]), attribute) == 17
    with pytest.raises(SystemExit):
        script.parse_args(required + [option, "17", "--seed", "17"])
