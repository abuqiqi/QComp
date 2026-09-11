"""验证标准结果的原子持久化、配置往返和中断生命周期。

主要内容：
- 结果校验及写入失败时原文件保持有效。
- 小型真实压缩在评测失败、中断或报告失败时的状态边界。
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_layer_selection_dashboard import make_sources
from torch import nn

from qcomp import (
    CompressionExecutionConfig,
    CompressionTarget,
    ModelLoadConfig,
    MPOSpec,
)
from qcomp.evaluation import (
    EvaluationResult,
    EvaluationTask,
    EvaluationTaskConfig,
    LMEvalConfig,
    evaluation_task_config_from_dict,
    evaluation_task_config_to_dict,
)
from qcomp.workflows import SensitivityExperimentConfig, run_sensitivity_experiment
from qcomp.workflows.sensitivity_io import (
    read_sensitivity_results,
    write_sensitivity_results,
)


def test_evaluation_config_roundtrip_and_missing_fields():
    """完整配置保留 null 和自定义 task，缺失字段不得悄悄使用默认值。"""
    config = EvaluationTaskConfig(
        LMEvalConfig("custom", num_fewshot=None, sample_start_index=7),
        {"ERROR_RATE": "lower"},
    )
    encoded = evaluation_task_config_to_dict(config)
    assert evaluation_task_config_from_dict(json.loads(json.dumps(encoded))) == config
    encoded["evaluation"].pop("evaluation_seed")
    with pytest.raises(ValueError, match="完整"):
        evaluation_task_config_from_dict(encoded)


def test_atomic_failure_preserves_previous_result(tmp_path):
    """原子替换失败时保留之前的字节和有效结果，并清理临时文件。"""
    config = make_sources(tmp_path, count=1, modules=1)
    path = tmp_path / config["datasets"][0]["path"]
    before = path.read_bytes()
    data = read_sensitivity_results(path)
    with (
        patch(
            "qcomp.workflows.sensitivity_io.os.replace",
            side_effect=OSError("disk failure"),
        ),
        pytest.raises(OSError),
    ):
        write_sensitivity_results(path, data)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]
    data["cases"][0]["metrics"]["acc"] = float("nan")
    with pytest.raises(ValueError):
        write_sensitivity_results(path, data)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("evaluation failed"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_case_progress_and_model_restore_on_failure(tmp_path, error, status):
    """第二个 case 评测失败时保存基线与第一个 case，原模型恢复且页面拒绝读取。"""
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(4))
    original = tuple(model)
    result_path = tmp_path / "sensitivity_results.json"
    snapshots = []
    task = EvaluationTask("custom", "synthetic", "test", "fixed", ("acc",))

    def evaluator(current):
        """按调用次序检查前一次已持久化进度，然后在第二层评测时失败。"""
        del current
        snapshots.append(json.loads(result_path.read_text()))
        if len(snapshots) == 3:
            raise error
        return EvaluationResult(task, {"acc": 0.75}, 2, total_examples=10)

    config = SensitivityExperimentConfig(
        name="test",
        evaluation=EvaluationTaskConfig(LMEvalConfig("custom"), {"acc": "higher"}),
        model=ModelLoadConfig(
            model_name_or_path="test", device="cpu", dtype=torch.float32
        ),
        compression=CompressionExecutionConfig(
            decomposition_provider="native", execution_provider="native"
        ),
        output=tmp_path / "report.md",
    )
    with (
        patch(
            "qcomp.workflows.sensitivity.load_causal_lm",
            return_value=SimpleNamespace(model=model, tokenizer=None),
        ),
        patch("qcomp.workflows.sensitivity.LMEvalEvaluator", return_value=evaluator),
        pytest.raises(type(error)),
    ):
        run_sensitivity_experiment(
            config,
            select_linear=lambda p, m: True,
            make_target=lambda p, m: CompressionTarget(
                p, "mpo", MPOSpec((2, 2), (2, 2), (1, 2, 1))
            ),
        )
    assert snapshots[0]["baseline"] is None
    assert snapshots[1]["baseline"]["metrics"] == {"acc": 0.75}
    assert len(snapshots[2]["cases"]) == 1
    document = read_sensitivity_results(result_path, completed=False)
    assert document["status"] == status and len(document["cases"]) == 1
    assert all(a is b for a, b in zip(model, original))
    assert not list(tmp_path.glob("*.jsonl"))
    with pytest.raises(ValueError, match="尚未完成"):
        read_sensitivity_results(result_path)


def test_report_failure_keeps_completed_results(tmp_path):
    """报告失败不能把已完成且可读取的实验改成 failed。"""
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    evaluation = EvaluationResult(
        EvaluationTask("custom", "synthetic", "test", "fixed", ("acc",)),
        {"acc": 0.75},
        2,
    )
    config = SensitivityExperimentConfig(
        name="test",
        evaluation=EvaluationTaskConfig(LMEvalConfig("custom"), {"acc": "higher"}),
        model=ModelLoadConfig(
            model_name_or_path="test", device="cpu", dtype=torch.float32
        ),
        compression=CompressionExecutionConfig(
            decomposition_provider="native", execution_provider="native"
        ),
        output=tmp_path / "report.md",
    )
    with (
        patch(
            "qcomp.workflows.sensitivity.load_causal_lm",
            return_value=SimpleNamespace(model=model, tokenizer=None),
        ),
        patch(
            "qcomp.workflows.sensitivity.LMEvalEvaluator",
            return_value=lambda m: evaluation,
        ),
        patch(
            "qcomp.workflows.sensitivity.format_sensitivity_report",
            side_effect=OSError("report failed"),
        ),
        pytest.raises(OSError, match="report failed"),
    ):
        run_sensitivity_experiment(
            config,
            select_linear=lambda p, m: True,
            make_target=lambda p, m: CompressionTarget(
                p, "mpo", MPOSpec((2,), (2,), (1, 1))
            ),
        )
    assert (
        read_sensitivity_results(tmp_path / "sensitivity_results.json")["status"]
        == "completed"
    )


def test_evaluator_initialization_failure_is_recorded(tmp_path):
    """计划已确定后，任务初始化失败也保存 failed 状态和空 case 列表。"""
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    config = SensitivityExperimentConfig(
        name="initialization",
        evaluation=EvaluationTaskConfig(LMEvalConfig("custom"), {"acc": "higher"}),
        model=ModelLoadConfig(
            model_name_or_path="test", device="cpu", dtype=torch.float32
        ),
        compression=CompressionExecutionConfig(
            decomposition_provider="native", execution_provider="native"
        ),
        output=tmp_path / "report.md",
    )
    with (
        patch(
            "qcomp.workflows.sensitivity.load_causal_lm",
            return_value=SimpleNamespace(model=model, tokenizer=None),
        ),
        patch(
            "qcomp.workflows.sensitivity.LMEvalEvaluator",
            side_effect=RuntimeError("task initialization"),
        ),
        pytest.raises(RuntimeError, match="task initialization"),
    ):
        run_sensitivity_experiment(
            config,
            select_linear=lambda p, m: True,
            make_target=lambda p, m: CompressionTarget(
                p, "mpo", MPOSpec((2,), (2,), (1, 1))
            ),
        )
    document = read_sensitivity_results(
        tmp_path / "sensitivity_results.json", completed=False
    )
    assert (
        document["status"] == "failed"
        and document["baseline"] is None
        and not document["cases"]
    )
