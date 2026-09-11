"""用确定性小矩阵验证 JSON 压缩脚本，模型加载和任务评测使用本地替身。

主要内容：
- 真正执行 native MPO 分解，验证 artifact、报告、前后评测及模块恢复。
- 验证 dry-run、跳过评测、输出保护及失败日志，不使用 GPU 或外部数据。
"""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qcomp import load_artifact, reconstruct_tensor
from qcomp.evaluation import EvaluationResult, EvaluationTask
from scripts import run_compression_plan as script


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """准备两种 rank 的小模型和一个有明确来源设置的评测任务。"""
    model = nn.ModuleDict(
        {name: nn.Linear(4, 4, bias=False) for name in ("first", "second")}
    )
    with torch.no_grad():
        for index, layer in enumerate(model.values()):
            layer.weight.copy_(torch.arange(16).reshape(4, 4) / 16 + index)
    targets = [
        dict(
            module_path=name,
            representation="mpo",
            spec=dict(out_modes=[2, 2], in_modes=[2, 2], ranks=[1, rank, 1]),
        )
        for name, rank in [("second", 2), ("first", 1)]
    ]
    config = dict(
        task="boolq",
        num_fewshot=None,
        limit=4,
        seed=17,
        batch_size=2,
        max_length=64,
        sample_start_index=3,
        apply_chat_template=False,
        metrics=["acc"],
        metric_directions={"acc": "higher"},
    )
    document = dict(
        kind="qcomp_compression_selection",
        model={"name_or_path": "fixture-model"},
        compression_plan={"targets": targets},
        selection={
            "settings": {
                "datasets": [{"id": "boolq", "enabled": True, "metric": "acc"}]
            },
            "sources": [
                {
                    "dataset_id": "boolq",
                    "provenance": {"configuration": config},
                    "path": "/missing/events.jsonl",
                }
            ],
        },
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document))
    calls, configs, outputs = [], [], []
    monkeypatch.setattr(
        script,
        "load_runtime_config",
        lambda _: SimpleNamespace(model_name_or_path="runtime-model", offline=True),
    )
    monkeypatch.setattr(script, "capture_console", lambda _: nullcontext())

    def load(config, **kwargs):
        """记录模型加载调用并返回固定小模型。"""
        calls.append(config)
        return SimpleNamespace(model=model, tokenizer=object())

    class Evaluator:
        """记录复用配置，在真实压缩层上前向并返回确定性得分。"""

        def __init__(self, tokenizer, config, **kwargs):
            """保存传入配置，确保前后复用同一实例。"""
            configs.append(config)
            self.calls = 0

        def __call__(self, current):
            """每阶段执行前向并输出指标及固定题数。"""
            self.calls += 1
            outputs.append(current["first"](torch.ones(1, 4)).detach().clone())
            return EvaluationResult(
                task=EvaluationTask("boolq", "synthetic", "test", "fixed", ("acc",)),
                metrics={"acc": 0.75 if self.calls == 1 else 0.5},
                evaluated_examples=4,
                total_examples=10,
            )

    monkeypatch.setattr(script, "load_causal_lm", load)
    monkeypatch.setattr(script, "LMEvalEvaluator", Evaluator)
    argv = [
        "--selection-json",
        str(path),
        "--output",
        str(tmp_path / "output"),
        "--device",
        "cpu",
        "--model-dtype",
        "float32",
        "--no-plot",
        "--decomposition-provider",
        "native",
        "--execution-provider",
        "native",
    ]
    return SimpleNamespace(
        model=model,
        path=path,
        document=document,
        argv=argv,
        root=tmp_path / "output",
        calls=calls,
        configs=configs,
        outputs=outputs,
    )


def test_joint_compression_and_saved_artifacts(experiment):
    """真实联合分解后保存的 artifact 可重建前向结果，并完整保留原模型。"""
    e = experiment
    originals = dict(e.model.items())
    weights = {name: layer.weight.clone() for name, layer in e.model.items()}
    source = e.path.read_bytes()
    script.main(e.argv)
    assert len(e.calls) == 1 and e.calls[0].model_name_or_path == "fixture-model"
    assert len(e.configs) == 1
    assert e.configs[0].num_fewshot is None and e.configs[0].sample_start_index == 3
    summary = json.loads((e.root / "summary.json").read_text())
    assert summary["compression"]["compressed_layers"] == 2
    assert summary["comparison"]["boolq"]["acc"]["degradation"] == 0.25
    assert summary["evaluation_status"] == "completed"
    assert (e.root / "selection.json").read_bytes() == source == e.path.read_bytes()
    artifacts = summary["artifacts"]
    assert [a["module_path"] for a in artifacts] == ["second", "first"]
    saved_weight = reconstruct_tensor(load_artifact(e.root / artifacts[1]["path"]))
    assert torch.allclose(torch.ones(1, 4) @ saved_weight.T, e.outputs[1], atol=1e-6)
    assert "指标原单位" in (e.root / "report.md").read_text()
    for name, layer in e.model.items():
        assert layer is originals[name] and torch.equal(layer.weight, weights[name])
    events = [
        json.loads(s)["event"]
        for s in (e.root / "events.jsonl").read_text().splitlines()
    ]
    assert (
        events[-1] == "experiment_completed"
        and events.count("evaluation_completed") == 2
    )


def test_dry_run_and_overrides(experiment, capsys):
    """dry-run 不创建目录或加载模型，展示生效的模型和评测覆盖。"""
    e = experiment
    script.main(
        e.argv
        + [
            "--dry-run",
            "--model",
            "override-model",
            "--eval-limit",
            "1",
            "--eval-batch-size",
            "3",
        ]
    )
    text = capsys.readouterr().out
    assert '"model": "override-model"' in text and '"limit": 1' in text
    assert not e.root.exists() and not e.calls
    args = script.parse_args(e.argv + ["--eval-limit", "1"])
    configs, _ = script.evaluation_configs(e.document, args)
    assert configs["boolq"].limit == 1 and configs["boolq"].sample_start_index == 3


def test_skip_evaluation_without_sources(experiment):
    """执行部分不依赖敏感度文件，关闭评测时允许省略分析内容。"""
    e = experiment
    del e.document["selection"]
    e.path.write_text(json.dumps(e.document))
    script.main(e.argv + ["--skip-eval"])
    summary = json.loads((e.root / "summary.json").read_text())
    assert summary["evaluation_status"] == "skipped" and summary["comparison"] == {}
    assert not e.configs and not e.outputs
    assert "跳过评测" in (e.root / "report.md").read_text()


def test_evaluation_failure_preserves_artifacts_and_restores(experiment, monkeypatch):
    """联合评测失败时保留已保存产物，写失败事件且不伪报成功。"""
    e = experiment
    originals = dict(e.model.items())
    base = script.LMEvalEvaluator

    class FailingEvaluator(base):
        """在真实联合压缩后的评测中注入错误。"""

        def __call__(self, model):
            """第一次为 baseline，第二次失败。"""
            if self.calls:
                raise RuntimeError("fixture failure")
            return super().__call__(model)

    monkeypatch.setattr(script, "LMEvalEvaluator", FailingEvaluator)
    with pytest.raises(RuntimeError, match="fixture failure"):
        script.main(e.argv)
    assert (e.root / "artifacts.json").is_file()
    assert not (e.root / "summary.json").exists()
    last = json.loads((e.root / "events.jsonl").read_text().splitlines()[-1])
    assert last["event"] == "experiment_failed"
    assert all(e.model[name] is layer for name, layer in originals.items())


def test_invalid_input_before_loading(experiment):
    """拒绝已有输出目录及不可验证的评测来源，避免启动昂贵加载。"""
    e = experiment
    e.root.mkdir()
    with pytest.raises(FileExistsError):
        script.main(e.argv)
    assert not e.calls
    e.root.rmdir()
    del e.document["selection"]["sources"][0]["provenance"]["configuration"][
        "num_fewshot"
    ]
    e.path.write_text(json.dumps(e.document))
    with pytest.raises(ValueError, match="评测配置"):
        script.main(e.argv)
    assert not e.root.exists() and not e.calls


@pytest.mark.parametrize(
    "extra",
    [
        ["--eval-limit", "0"],
        ["--eval-batch-size", "-1"],
        ["--skip-eval", "--eval-limit", "1"],
    ],
)
def test_invalid_arguments(experiment, extra):
    """无效或互相冲突的命令行参数在执行前拒绝。"""
    with pytest.raises(SystemExit):
        script.parse_args(experiment.argv + extra)


def test_target_mismatch_stops_before_evaluation(experiment):
    """真实模型目标不匹配时在 baseline 之前失败，并留下明确事件。"""
    e = experiment
    e.document["compression_plan"]["targets"][0]["module_path"] = "missing"
    e.path.write_text(json.dumps(e.document))
    with pytest.raises(ValueError, match="missing"):
        script.main(e.argv)
    assert not e.configs and not e.outputs
    assert not (e.root / "artifacts.json").exists()
    last = json.loads((e.root / "events.jsonl").read_text().splitlines()[-1])
    assert last["event"] == "experiment_failed"


def test_independent_multi_task_metrics_and_png(experiment, monkeypatch):
    """独立任务完全替代来源列表，多指标结果和数值标注正确生成 PNG。"""
    e = experiment
    config_path = e.path.parent / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"task": "other", "metrics": ["acc", "loss"], "num_fewshot": 0},
                    {
                        "task": "custom",
                        "metrics": ["quality"],
                        "metric_directions": {"quality": "higher"},
                        "limit": 2,
                    },
                ]
            }
        )
    )
    seen = []

    class Evaluator:
        """为两个不同任务提供可验证的多指标结果。"""

        def __init__(self, tokenizer, config, **kwargs):
            """记录独立配置。"""
            self.config, self.calls = config, 0
            seen.append(config)

        def __call__(self, model):
            """同一实例在 baseline 与压缩模型上分别执行一次。"""
            self.calls += 1
            return EvaluationResult(
                EvaluationTask(
                    self.config.task,
                    "synthetic",
                    "test",
                    "fixed",
                    ("acc", "loss", "quality"),
                ),
                {
                    "acc": 0.8 if self.calls == 1 else 0.6,
                    "loss": 1.0 if self.calls == 1 else 1.2,
                    "quality": 2.0 if self.calls == 1 else 1.0,
                },
                2,
                total_examples=10,
            )

    monkeypatch.setattr(script, "LMEvalEvaluator", Evaluator)
    plt = script.plotting_module()
    captured = []
    original_close = plt.close

    def close(figure=None):
        """关闭图前读取柱标签、数值和各子图标题。"""
        if hasattr(figure, "axes"):
            captured.extend(
                (
                    ax.get_title(),
                    [p.get_height() for p in ax.patches],
                    [t.get_text() for t in ax.texts],
                )
                for ax in figure.axes
            )
        original_close(figure)

    monkeypatch.setattr(plt, "close", close)
    script.main(
        [arg for arg in e.argv if arg != "--no-plot"]
        + [
            "--eval-config",
            str(config_path),
            "--eval-limit",
            "3",
            "--eval-batch-size",
            "5",
        ]
    )
    assert [c.task for c in seen] == ["other", "custom"]
    assert all(c.limit == 3 and c.batch_size == 5 for c in seen)
    assert seen[0].num_fewshot == 0 and seen[1].num_fewshot is None
    summary = json.loads((e.root / "summary.json").read_text())
    assert set(summary["comparison"]) == {"other", "custom"}
    assert summary["comparison"]["other"]["loss"]["degradation"] == pytest.approx(0.2)
    assert (e.root / "scores.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(captured) == 3
    assert captured[0][1] == pytest.approx([0.8, 0.6])
    assert captured[0][2] == ["0.8", "0.6"]
    assert "lower is better" in captured[1][0]
    assert "loss" in (e.root / "report.md").read_text()


@pytest.mark.parametrize(
    "bad",
    [
        {"datasets": []},
        {"datasets": [{"task": "x", "metrics": []}]},
        {"datasets": [{"task": "x", "metrics": ["acc", "ACC"]}]},
        {"datasets": [{"task": "x", "metrics": ["unknown"]}]},
        {
            "datasets": [
                {"task": "x", "metrics": ["acc"], "metric_directions": {"acc": "bad"}}
            ]
        },
        {
            "datasets": [
                {"task": "x", "metrics": ["acc"]},
                {"task": "x", "metrics": ["loss"]},
            ]
        },
    ],
)
def test_invalid_independent_config_before_loading(experiment, bad):
    """独立配置错误在加载模型前拒绝。"""
    e = experiment
    path = e.path.parent / "eval.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        script.main(e.argv + ["--eval-config", str(path)])
    assert not e.calls and not e.root.exists()


def test_plotting_preflight_and_output_failure(experiment, monkeypatch):
    """缺少绘图依赖时不加载模型，绘图失败时保留实测结果并记录失败。"""
    e = experiment
    argv = [arg for arg in e.argv if arg != "--no-plot"]

    def unavailable():
        """模拟可选依赖未安装。"""
        raise ImportError("matplotlib unavailable")

    monkeypatch.setattr(script, "plotting_module", unavailable)
    with pytest.raises(ImportError):
        script.main(argv)
    assert not e.calls and not e.root.exists()
    monkeypatch.setattr(script, "plotting_module", lambda: None)

    def failed_plot(*args):
        """模拟实际绘图阶段失败。"""
        raise RuntimeError("plot failed")

    monkeypatch.setattr(script, "plot_scores", failed_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        script.main(argv)
    assert (e.root / "summary.json").exists() and (e.root / "report.md").exists()
    records = [
        json.loads(s) for s in (e.root / "events.jsonl").read_text().splitlines()
    ]
    assert records[-1]["event"] == "experiment_failed"
    assert records[-1]["fields"]["stage"] == "plotting"
    assert not any(r["event"] == "experiment_completed" for r in records)


def test_skip_eval_config_conflict_and_no_plot(experiment, monkeypatch):
    """关闭评测禁止独立配置；关闭绘图不访问 Matplotlib。"""
    e = experiment
    with pytest.raises(SystemExit):
        script.parse_args(e.argv + ["--skip-eval", "--eval-config", "ignored.json"])

    def forbidden():
        """不应检查绘图依赖。"""
        raise AssertionError("unexpected plotting dependency")

    monkeypatch.setattr(script, "plotting_module", forbidden)
    script.main(e.argv)
    assert not (e.root / "scores.png").exists()


@pytest.mark.parametrize("skip", [False, True])
def test_decomposition_seed_independent_of_evaluator(experiment, monkeypatch, skip):
    """任务评测与模型加载消耗随机数后，压缩仍从指定分解种子开始。"""
    import qcomp.workflows.evaluate as workflow

    e = experiment
    original = workflow.compress_model
    base = script.LMEvalEvaluator

    class RandomEvaluator(base):
        """模拟评测对全局随机状态的消耗。"""

        def __call__(self, model):
            """消耗随机数后继续确定性评测。"""
            torch.rand(7)
            return super().__call__(model)

    monkeypatch.setattr(script, "LMEvalEvaluator", RandomEvaluator)

    def compress(*args, **kwargs):
        """压缩入口前检查生成器状态。"""
        expected = torch.Generator().manual_seed(123).get_state()
        assert torch.equal(torch.get_rng_state(), expected)
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, "compress_model", compress)
    script.main(e.argv + ["--seed", "123"] + (["--skip-eval"] if skip else []))
