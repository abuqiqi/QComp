"""验证 baseline input 缓存和单目标局部输出 NMSE 工作流。"""

from pathlib import Path

import pytest
import torch
from torch import nn

from qcomp import CompressionPlan, CompressionTarget, MPOSpec, get_backend
from qcomp.workflows import (
    capture_local_output_inputs,
    capture_local_output_inputs_in_memory,
    evaluate_local_output_nmse_plans,
    evaluate_local_output_nmse_plans_in_memory,
    iter_local_output_input_batches,
    local_output_cache_fingerprint,
    validate_local_output_input_cache,
)


class TinyCausalLM(nn.Module):
    """提供可计数完整前向和一个目标 Linear 的确定性小模型。"""

    def __init__(self) -> None:
        """建立 4 维 embedding 与无 bias 投影。"""
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 4, bias=False)
        self.forward_calls = 0
        torch.manual_seed(7)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(32).reshape(8, 4) / 10)
            self.projection.weight.copy_(torch.arange(16).reshape(4, 4) / 8)

    def forward(self, *, input_ids, attention_mask, use_cache):
        """执行目标投影并记录完整模型调用次数。"""
        del attention_mask, use_cache
        self.forward_calls += 1
        return self.projection(self.embedding(input_ids))


class RepeatedProjectionLM(TinyCausalLM):
    """在同一 batch 中两次调用目标层，用于触发采集失败。"""

    def forward(self, *, input_ids, attention_mask, use_cache):
        """连续执行两次目标投影。"""
        del attention_mask, use_cache
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        self.projection(hidden)
        return self.projection(hidden)


class SharedProjectionLM(nn.Module):
    """让 attention 和 MLP 投影分别共享输入，用于验证 storage 去重。"""

    def __init__(self) -> None:
        """建立 embedding 和五个同形状投影。"""
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        """让 q/k/v 共享输入，并让 gate/up 共享另一份输入。"""
        del attention_mask, use_cache
        self.forward_calls += 1
        attention_hidden = self.embedding(input_ids)
        mlp_hidden = attention_hidden.clone()
        return (
            self.q_proj(attention_hidden)
            + self.k_proj(attention_hidden)
            + self.v_proj(attention_hidden)
            + self.gate_proj(mlp_hidden)
            + self.up_proj(mlp_hidden)
        )


def batches():
    """返回两个 batch，其中第二个包含一个 padding token。"""
    return [
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])},
        {"input_ids": torch.tensor([[3, 0]]), "attention_mask": torch.tensor([[1, 0]])},
    ]


def cache_identity():
    """返回测试共享的模型与校准配置身份。"""
    return {"model": "tiny", "tokenizer": "tiny"}, {"datasets": ["fixed"], "seed": 3}


def test_capture_validate_and_stream_without_extra_dense_forward(
    tmp_path: Path,
) -> None:
    """dense 数据只遍历一次，缓存可验证并供候选直接执行目标层。"""
    model = TinyCausalLM()
    model.train()
    model_identity, calibration = cache_identity()
    cache = tmp_path / "cache"
    capture_local_output_inputs(
        model,
        {"fixed": batches()},
        ("projection",),
        cache,
        model_identity=model_identity,
        calibration_config=calibration,
    )
    assert model.forward_calls == 2
    assert model.training
    assert not model.projection._forward_pre_hooks

    manifest = validate_local_output_input_cache(
        cache,
        expected_model_identity=model_identity,
        expected_calibration_config=calibration,
        required_module_paths=("projection",),
    )
    loaded = list(
        iter_local_output_input_batches(
            cache, manifest, source="fixed", module_path="projection"
        )
    )
    assert len(loaded) == 2
    assert loaded[1][1].tolist() == [[1, 0]]

    original = model.projection
    plan = CompressionPlan(
        (CompressionTarget("projection", "mpo", MPOSpec((2, 2), (2, 2), (1, 1, 1))),)
    )
    backend = get_backend("native", "mpo")
    result = evaluate_local_output_nmse_plans(
        model,
        {"projection-rank-1": plan},
        cache_path=cache,
        manifest=manifest,
        decomposition_backends={"mpo": backend},
        execution_backends={"mpo": backend},
        decomposition_dtype=torch.float64,
    )
    assert model.forward_calls == 2
    assert model.projection is original
    assert (
        result.plan_results[0].evaluations["fixed"].evaluation.metrics["output_nmse"]
        > 0
    )
    assert result.plan_results[0].evaluations["fixed"].evaluation.evaluated_tokens == 3


def test_cache_identity_is_stable_and_mismatch_is_rejected(tmp_path: Path) -> None:
    """同一身份得到稳定 fingerprint，任何校准配置变化都会拒绝复用。"""
    model_identity, calibration = cache_identity()
    first = local_output_cache_fingerprint(
        model_identity=model_identity,
        calibration_config=calibration,
        module_paths=("projection",),
    )
    second = local_output_cache_fingerprint(
        model_identity={"tokenizer": "tiny", "model": "tiny"},
        calibration_config={"seed": 3, "datasets": ["fixed"]},
        module_paths=("projection",),
    )
    assert first == second
    cache = tmp_path / "cache"
    capture_local_output_inputs(
        TinyCausalLM(),
        {"fixed": batches()},
        ("projection",),
        cache,
        model_identity=model_identity,
        calibration_config=calibration,
    )
    with pytest.raises(ValueError, match="identity"):
        validate_local_output_input_cache(
            cache,
            expected_model_identity=model_identity,
            expected_calibration_config={**calibration, "seed": 4},
            required_module_paths=("projection",),
        )


def test_workflow_requires_one_target_per_plan(tmp_path: Path) -> None:
    """局部阶段拒绝把两个投影装进同一个压缩计划。"""
    model = TinyCausalLM()
    model_identity, calibration = cache_identity()
    cache = tmp_path / "cache"
    capture_local_output_inputs(
        model,
        {"fixed": batches()},
        ("projection",),
        cache,
        model_identity=model_identity,
        calibration_config=calibration,
    )
    manifest = validate_local_output_input_cache(
        cache,
        expected_model_identity=model_identity,
        expected_calibration_config=calibration,
        required_module_paths=("projection",),
    )
    spec = MPOSpec((2, 2), (2, 2), (1, 1, 1))
    plan = CompressionPlan(
        (
            CompressionTarget("projection", "mpo", spec),
            CompressionTarget("embedding", "mpo", spec),
        )
    )
    backend = get_backend("native", "mpo")
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_local_output_nmse_plans(
            model,
            {"invalid": plan},
            cache_path=cache,
            manifest=manifest,
            decomposition_backends={"mpo": backend},
            execution_backends={"mpo": backend},
        )


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    """缓存分片不完整时必须报错，不得静默重用。"""
    model_identity, calibration = cache_identity()
    cache = tmp_path / "cache"
    capture_local_output_inputs(
        TinyCausalLM(),
        {"fixed": batches()},
        ("projection",),
        cache,
        model_identity=model_identity,
        calibration_config=calibration,
    )
    (cache / "masks" / "000-00000.pt").unlink()
    with pytest.raises(FileNotFoundError, match="mask shard"):
        validate_local_output_input_cache(
            cache,
            expected_model_identity=model_identity,
            expected_calibration_config=calibration,
            required_module_paths=("projection",),
        )


def test_capture_failure_removes_hooks_and_temporary_cache(tmp_path: Path) -> None:
    """采集中途失败时恢复训练状态并删除未发布的缓存。"""
    model = RepeatedProjectionLM()
    model.train()
    model_identity, calibration = cache_identity()
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="more than once"):
        capture_local_output_inputs(
            model,
            {"fixed": batches()},
            ("projection",),
            cache,
            model_identity=model_identity,
            calibration_config=calibration,
        )
    assert model.training
    assert not model.projection._forward_pre_hooks
    assert not cache.exists()
    assert list(tmp_path.iterdir()) == []


def test_memory_capture_stays_on_model_device_and_deduplicates_storage(
    tmp_path: Path,
) -> None:
    """内存捕获不落盘，且共享输入的投影只保留一份底层 storage。"""
    model = SharedProjectionLM()
    model.train()
    paths = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    inputs, masks, records = capture_local_output_inputs_in_memory(
        model, {"fixed": batches()}, paths
    )
    assert model.forward_calls == 2
    assert model.training
    assert all(not getattr(model, path)._forward_pre_hooks for path in paths)
    assert list(tmp_path.iterdir()) == []
    for batch_index in range(2):
        attention_tensors = [
            inputs[("fixed", path)][batch_index]
            for path in ("q_proj", "k_proj", "v_proj")
        ]
        mlp_tensors = [
            inputs[("fixed", path)][batch_index] for path in ("gate_proj", "up_proj")
        ]
        attention_pointers = {
            tensor.untyped_storage().data_ptr() for tensor in attention_tensors
        }
        mlp_pointers = {tensor.untyped_storage().data_ptr() for tensor in mlp_tensors}
        assert len(attention_pointers) == len(mlp_pointers) == 1
        assert attention_pointers.isdisjoint(mlp_pointers)
        tensors = attention_tensors + mlp_tensors
        assert all(
            tensor.device == next(model.parameters()).device for tensor in tensors
        )
        assert masks["fixed"][batch_index].device == next(model.parameters()).device
    expected_bytes = sum(
        2
        * inputs[("fixed", "q_proj")][index].numel()
        * inputs[("fixed", "q_proj")][index].element_size()
        + masks["fixed"][index].numel() * masks["fixed"][index].element_size()
        for index in range(2)
    )
    assert records["fixed"]["input_tensor_bytes"] == expected_bytes


def test_memory_and_disk_evaluation_match(tmp_path: Path) -> None:
    """同一小模型输入的内存和磁盘入口产生相同局部输出统计。"""
    model = TinyCausalLM()
    model_identity, calibration = cache_identity()
    cache = tmp_path / "cache"
    capture_local_output_inputs(
        model,
        {"fixed": batches()},
        ("projection",),
        cache,
        model_identity=model_identity,
        calibration_config=calibration,
    )
    manifest = validate_local_output_input_cache(
        cache,
        expected_model_identity=model_identity,
        expected_calibration_config=calibration,
        required_module_paths=("projection",),
    )
    inputs, masks, records = capture_local_output_inputs_in_memory(
        model, {"fixed": batches()}, ("projection",)
    )
    plan = CompressionPlan(
        (CompressionTarget("projection", "mpo", MPOSpec((2, 2), (2, 2), (1, 1, 1))),)
    )
    backend = get_backend("native", "mpo")
    arguments = {
        "decomposition_backends": {"mpo": backend},
        "execution_backends": {"mpo": backend},
        "decomposition_dtype": torch.float64,
    }
    disk = evaluate_local_output_nmse_plans(
        model,
        {"projection-rank-1": plan},
        cache_path=cache,
        manifest=manifest,
        **arguments,
    )
    memory = evaluate_local_output_nmse_plans_in_memory(
        model,
        {"projection-rank-1": plan},
        inputs=inputs,
        masks=masks,
        source_records=records,
        **arguments,
    )
    disk_metrics = disk.plan_results[0].evaluations["fixed"].evaluation.metrics
    memory_metrics = memory.plan_results[0].evaluations["fixed"].evaluation.metrics
    assert memory_metrics == pytest.approx(disk_metrics)
    assert model.forward_calls == 4


def test_memory_capture_failure_restores_model_state() -> None:
    """内存捕获失败时移除 hook 并恢复模型训练状态。"""
    model = RepeatedProjectionLM()
    model.train()
    with pytest.raises(ValueError, match="more than once"):
        capture_local_output_inputs_in_memory(
            model, {"fixed": batches()}, ("projection",)
        )
    assert model.training
    assert not model.projection._forward_pre_hooks


def test_memory_evaluation_failure_restores_original_linear() -> None:
    """候选评测遇到损坏 mask 时仍恢复原始 Linear 和模型训练状态。"""
    model = TinyCausalLM()
    model.train()
    original = model.projection
    values, masks, records = capture_local_output_inputs_in_memory(
        model, {"fixed": batches()}, ("projection",)
    )
    masks["fixed"][0] = torch.ones(1, 1, dtype=torch.long)
    plan = CompressionPlan(
        (CompressionTarget("projection", "mpo", MPOSpec((2, 2), (2, 2), (1, 1, 1))),)
    )
    backend = get_backend("native", "mpo")
    with pytest.raises(ValueError, match="valid mask"):
        evaluate_local_output_nmse_plans_in_memory(
            model,
            {"projection-rank-1": plan},
            inputs=values,
            masks=masks,
            source_records=records,
            decomposition_backends={"mpo": backend},
            execution_backends={"mpo": backend},
            decomposition_dtype=torch.float64,
        )
    assert model.projection is original
    assert model.training
