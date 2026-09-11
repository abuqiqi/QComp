"""验证多方案、多任务压缩评测及所有恢复路径，使用确定性 CPU 小模型。

主要内容：
- 共享基线、多个指标方向、逐层统计、回调顺序和无张量结果。
- 分解、统计、评测和回调错误均停止后续方案并恢复模型。
- 输入、任务与样本范围校验及 BF16/FP32 精度传递。
"""

from dataclasses import fields, is_dataclass, replace
import inspect
from collections.abc import Mapping

import pytest
import torch
from torch import nn

from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    evaluate_compression_plans,
    get_backend,
    list_tensor_network_linears,
)
from qcomp.evaluation import EvaluationResult, EvaluationTask
import qcomp.workflows.evaluate as workflow


@pytest.fixture
def setup():
    """创建两个固定权重 Linear、两个不同目标集合以及 native 后端。"""
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    ).double()
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(4))
    model.train()
    model[1].eval()
    spec = MPOSpec.full_rank((2, 2), (2, 2))
    plans = {
        name: CompressionPlan(tuple(CompressionTarget(p, "mpo", spec) for p in paths))
        for name, paths in (("one", ["0"]), ("both", ["0", "1"]))
    }
    backend = get_backend("native", "mpo")
    return (
        model,
        plans,
        dict(
            decomposition_backends={"mpo": backend}, execution_backends={"mpo": backend}
        ),
    )


def evaluation(model, task="quality"):
    """根据压缩层数量返回确定性的质量、损失和附加指标。"""
    n = len(list_tensor_network_linears(model))
    return EvaluationResult(
        EvaluationTask(task, "synthetic", "test", "fixed", ("acc", "loss", "extra")),
        {"acc": 0.8 - 0.1 * n, "loss": 1 + 0.2 * n, "extra": 10 + n},
        4,
        total_examples=10,
    )


def assert_no_tensors(value):
    """递归检查返回结构不含模型、张量或替换记录。"""
    assert not isinstance(value, (torch.Tensor, nn.Module))
    if is_dataclass(value):
        for field in fields(value):
            assert_no_tensors(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            assert_no_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_tensors(item)


def test_multiple_plans_tasks_metrics_callbacks(setup, monkeypatch):
    """两方案两任务共六次评测，压缩两次，回调时序与逐层统计正确。"""
    model, plans, kwargs = setup
    originals = list(model)
    events, restored, compressions = [], [], []
    compress = workflow.compress_model

    def tracked(*args, **options):
        """确认每个方案都从原始模型开始，实际调用公共压缩接口。"""
        assert list(model) == originals
        compressions.append(args[1])
        return compress(*args, **options)

    monkeypatch.setattr(workflow, "compress_model", tracked)

    def evaluate(name):
        """为不同任务返回 evaluator，检查推理状态。"""

        def run(current):
            """记录模型状态并生成该任务结果。"""
            assert not any(m.training for m in current.modules())
            events.append(("eval", name, len(list_tensor_network_linears(current))))
            return evaluation(current, name)

        return run

    def compressed(name, artifacts, stats, seconds):
        """压缩通知早于任何误差统计和该方案任务评测。"""
        events.append(("compressed", name))
        assert len(artifacts) == stats.compressed_layers
        assert seconds >= 0

    layer_metrics = workflow.compression_metrics

    def measured(*args):
        """记录每个权重误差计算，不增加任务评测。"""
        events.append(("layer",))
        return layer_metrics(*args)

    monkeypatch.setattr(workflow, "compression_metrics", measured)

    def completed(result):
        """方案回调看到原 Linear 及原始混合训练状态。"""
        assert list(model) == originals
        assert model.training and model[0].training and not model[1].training
        restored.append(result)
        events.append(("done", result.name))

    result = evaluate_compression_plans(
        model,
        plans,
        evaluators={t: evaluate(t) for t in ("a", "b")},
        metric_directions={t: {"acc": "higher", "loss": "lower"} for t in ("a", "b")},
        collect_layer_metrics=True,
        on_compressed=compressed,
        on_plan_result=completed,
        on_evaluation=lambda plan, task, timed: events.append(("notified", plan, task)),
        **kwargs,
    )
    assert len(compressions) == 2
    assert [e for e in events if e[0] == "eval"] == [
        ("eval", t, n) for n in (0, 1, 2) for t in ("a", "b")
    ]
    assert [e for e in events if e[0] in ("compressed", "layer", "done")] == [
        ("compressed", "one"),
        ("layer",),
        ("done", "one"),
        ("compressed", "both"),
        ("layer",),
        ("layer",),
        ("done", "both"),
    ]
    assert tuple(restored) == result.plan_results
    for index, item in enumerate(result.plan_results, 1):
        assert len(item.layer_compressions) == index
        assert item.metric_degradations["a"] == pytest.approx(
            {"acc": 0.1 * index, "loss": 0.2 * index}
        )
        assert item.evaluations["b"].seconds >= 0
    assert result.baseline["a"].evaluation.metrics["extra"] == 10
    assert_no_tensors(result)


@pytest.mark.parametrize(
    "failure",
    [
        "baseline",
        "evaluation",
        "compressed_callback",
        "evaluation_callback",
        "plan_callback",
        "layer_metrics",
        "decomposition",
    ],
)
def test_failures_restore_all_states(setup, monkeypatch, failure):
    """错误不会继续后续方案，所有原模块、权重和训练状态均恢复。"""
    model, plans, kwargs = setup
    originals, weights = list(model), [m.weight.clone() for m in model]

    def fail(*args, **kw):
        """注入确定性异常。"""
        raise RuntimeError("planned failure")

    def evaluator(current):
        """按要求在基线或联合评测时失败。"""
        if failure == "baseline" or (
            failure == "evaluation" and list_tensor_network_linears(current)
        ):
            fail()
        return evaluation(current)

    options = {}
    if failure == "compressed_callback":
        options["on_compressed"] = fail
    if failure == "evaluation_callback":
        options["on_evaluation"] = fail
    if failure == "plan_callback":
        options["on_plan_result"] = fail
    if failure == "layer_metrics":
        options["collect_layer_metrics"] = True
        monkeypatch.setattr(workflow, "compression_metrics", fail)
    if failure == "decomposition":
        original = kwargs["decomposition_backends"]["mpo"].decompose
        calls = []

        def decompose(*args):
            """第二层失败，覆盖部分替换后的回滚。"""
            calls.append(1)
            if len(calls) == 2:
                fail()
            return original(*args)

        monkeypatch.setattr(
            kwargs["decomposition_backends"]["mpo"], "decompose", decompose
        )
        plans = {"both": plans["both"]}
    with pytest.raises(RuntimeError, match="planned failure"):
        evaluate_compression_plans(
            model,
            plans,
            evaluators={"task": evaluator},
            metric_directions={"task": {"acc": "higher"}},
            **options,
            **kwargs,
        )
    assert list(model) == originals
    assert model.training and model[0].training and not model[1].training
    for current, before in zip(model, weights):
        assert torch.equal(current.weight, before)


@pytest.mark.parametrize("change", ["task", "count", "total", "missing", "nan"])
def test_result_validation(setup, change):
    """联合评测必须保持任务范围并返回有限关注指标。"""
    model, plans, kwargs = setup

    def evaluator(current):
        """仅在压缩状态注入无效结果。"""
        result = evaluation(current)
        if not list_tensor_network_linears(current):
            return result
        if change == "task":
            return replace(result, task=replace(result.task, preprocessing="other"))
        if change == "count":
            return replace(result, evaluated_examples=3)
        if change == "total":
            return replace(result, total_examples=11)
        if change == "missing":
            return replace(result, metrics={})
        return replace(result, metrics={"acc": float("nan")})

    with pytest.raises(ValueError):
        evaluate_compression_plans(
            model,
            plans,
            evaluators={"task": evaluator},
            metric_directions={"task": {"acc": "higher"}},
            **kwargs,
        )
    assert all(isinstance(m, nn.Linear) for m in model)


def test_inputs_empty_evaluators_and_optional_metrics(setup, monkeypatch):
    """空 evaluator 合法且不重建权重，无效名称、方向和 backend 在评测前拒绝。"""
    model, plans, kwargs = setup

    def forbidden(*args):
        """关闭逐层统计时不应调用。"""
        raise AssertionError("unexpected reconstruction")

    monkeypatch.setattr(workflow, "compression_metrics", forbidden)
    result = evaluate_compression_plans(
        model, plans, evaluators={}, metric_directions={}, **kwargs
    )
    assert not result.baseline and all(
        not r.layer_compressions and not r.evaluations for r in result.plan_results
    )
    for bad_plans, directions in [
        ({}, {}),
        ({" ": plans["one"]}, {}),
        (plans, {"x": {}}),
        (plans, {"task": {}}),
        (plans, {"task": {"acc": "largest"}}),
        (plans, {"task": {"acc": "higher", " ACC ": "higher"}}),
    ]:
        with pytest.raises(ValueError):
            evaluate_compression_plans(
                model,
                bad_plans,
                evaluators={"task": evaluation},
                metric_directions=directions,
                **kwargs,
            )
    with pytest.raises(ValueError, match="backend"):
        evaluate_compression_plans(
            model,
            plans,
            evaluators={},
            metric_directions={},
            decomposition_backends={},
            execution_backends=kwargs["execution_backends"],
        )


def test_float32_decomposition_bfloat16_execution(setup, monkeypatch):
    """FP32 分解后执行精度仍为 BF16，输入输出与原权重保持。"""
    model, plans, kwargs = setup
    model.bfloat16()
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    expected = model(inputs).detach().clone()
    backend = kwargs["decomposition_backends"]["mpo"]
    original, dtypes = backend.decompose, []

    def decompose(weight, spec):
        """记录真实分解使用的精度。"""
        dtypes.append(weight.dtype)
        return original(weight, spec)

    monkeypatch.setattr(backend, "decompose", decompose)

    def evaluator(current):
        """检查压缩与未压缩状态的前向。"""
        assert all(p.dtype == torch.bfloat16 for p in current.parameters())
        torch.testing.assert_close(current(inputs), expected, rtol=0.05, atol=0.005)
        return evaluation(current)

    evaluate_compression_plans(
        model,
        {"both": plans["both"]},
        evaluators={"task": evaluator},
        metric_directions={"task": {"acc": "higher"}},
        decomposition_dtype=torch.float32,
        **kwargs,
    )
    assert dtypes == [torch.float32, torch.float32]


def test_no_resource_creation_dependency():
    """底层入口不创建 backend、spec 或具体数据集 evaluator。"""
    source = inspect.getsource(evaluate_compression_plans)
    for forbidden in ("get_backend", "MPOSpec", "tensorly", "LMEvalEvaluator"):
        assert forbidden not in source
