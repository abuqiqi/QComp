"""根据选层 JSON 联合压缩模型，评测前后得分并保存逐矩阵 artifact。

输入完整选层文件，复用 qcomp 的模型加载、计划校验、压缩、lm-eval 和存储接口；
默认评测 JSON 中勾选的任务，使用同一 evaluator 重测原模型和联合压缩模型。
输出输入快照、执行配置、artifact 清单、事件日志、summary.json 与 report.md；不微调。

主要内容：
- ``parse_args``、``evaluation_configs``：解析执行参数并还原来源中的评测设置。
- ``plot_scores``：用独立子图展示各任务、各指标的前后得分。
- ``write_report``：生成联合压缩报告。
- ``main``：串联加载、校验、压缩、保存和评测，异常时记录失败并恢复模型。

使用说明：
    python scripts/run_compression_plan.py --selection-json path/to/layer-selection.json
    添加 --dry-run 只检查文件与配置，不加载模型或创建产物目录。
    添加 --skip-eval 只压缩并保存；--eval-limit 仅用于临时覆盖各任务评测上限。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn

from qcomp import (
    ModelLoadConfig,
    evaluate_compression_plans,
    CompressionPlanEvaluation,
    TimedEvaluation,
    compression_plan_from_dict,
    CompressionExecutionConfig,
    load_causal_lm,
    load_compression_plan,
    load_runtime_config,
    log_event,
    save_artifact,
)
from qcomp.evaluation import (
    EvaluationTaskConfig,
    LMEvalConfig,
    LMEvalEvaluator,
    resolve_metric_directions,
)
from qcomp.logging import capture_console

PROJECT = Path(__file__).resolve().parents[1]
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 argv；目标和 ranks 只从 JSON 读取，拒绝无效评测上限与冲突参数。"""
    parser = argparse.ArgumentParser(
        description="按选层 JSON 联合压缩、评测并保存，不微调。", allow_abbrev=False
    )
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument(
        "--model", help="覆盖 JSON 模型来源；未提供时使用 JSON，再使用 runtime 配置"
    )
    parser.add_argument(
        "--runtime-config", help="运行配置 TOML，与现有脚本路径规则一致"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--decomposition-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--decomposition-provider", default="tensorly")
    parser.add_argument("--execution-provider", default="tensorly")
    parser.add_argument("--trust-remote-code", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument(
        "--decomposition-seed",
        type=int,
        default=42,
        help="分解随机种子；评测种子沿用各任务来源",
    )
    seeds.add_argument(
        "--seed",
        dest="decomposition_seed",
        type=int,
        default=argparse.SUPPRESS,
        help="兼容参数；请使用 --decomposition-seed",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help="覆盖每个任务的评测样本上限；group 为每个子任务上限",
    )
    parser.add_argument("--eval-batch-size", type=int, help="覆盖评测 batch size")
    parser.add_argument(
        "--eval-config", type=Path, help="独立的多任务评测 JSON，替代选层来源设置"
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成得分图")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="不加载模型，只检查文件并展示执行配置"
    )
    parser.add_argument("--output", type=Path, help="新产物目录，必须不存在")
    args = parser.parse_args(argv)
    for name in ("eval_limit", "eval_batch_size"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.skip_eval and (
        args.eval_limit is not None
        or args.eval_batch_size is not None
        or args.eval_config is not None
    ):
        parser.error("--skip-eval 不能与 --eval-config 或评测覆盖参数同时使用")
    return args


def evaluation_configs(
    document: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, EvaluationTaskConfig]:
    """从 JSON 的勾选项和 provenance 还原评测配置，不访问原敏感度文件。

    返回：
        按任务索引的完整评测配置。关闭评测时为空。
    """
    if args.skip_eval:
        return {}
    if args.eval_config is not None:
        configs = independent_evaluation_configs(args)
        return apply_evaluation_overrides(configs, args)
    selection = document.get("selection", {})
    datasets = selection.get("settings", {}).get("datasets")
    sources = selection.get("sources")
    if not isinstance(datasets, list) or not isinstance(sources, list):
        raise ValueError("JSON 缺少评测设置及来源；只压缩请使用 --skip-eval")
    source_map = {}
    for source in sources:
        key = source["dataset_id"]
        if key in source_map:
            raise ValueError(f"重复数据来源：{key}")
        source_map[key] = source
    configs = {}
    for item in datasets:
        if type(item.get("enabled")) is not bool:
            raise ValueError("数据集 enabled 必须为布尔值")
        if not item["enabled"]:
            continue
        task = item["id"]
        if task in configs:
            raise ValueError(f"重复评测任务：{task}")
        source = source_map.get(task, {}).get("provenance", {}).get("configuration", {})
        fields = (
            "task",
            "num_fewshot",
            "limit",
            "seed",
            "apply_chat_template",
            "batch_size",
            "max_length",
        )
        if any(field not in source for field in fields) or source["task"] != task:
            raise ValueError(f"{task}: 来源缺少可验证的评测配置")
        metric = item["metric"]
        direction = source.get("metric_directions", {}).get(metric)
        if direction not in ("higher", "lower") or metric not in source.get(
            "metrics", []
        ):
            raise ValueError(f"{task}: 无法确认指标 {metric} 及其方向")
        config = LMEvalConfig(
            **{
                ("evaluation_seed" if field == "seed" else field): source[field]
                for field in fields
            },
            sample_start_index=source.get("sample_start_index", 0),
        )
        configs[task] = EvaluationTaskConfig(config, {metric: direction})
    if not configs:
        raise ValueError("没有勾选的评测任务；只压缩请使用 --skip-eval")
    return apply_evaluation_overrides(configs, args)


def independent_evaluation_configs(
    args: argparse.Namespace,
) -> dict[str, EvaluationTaskConfig]:
    """读取独立多任务配置，规范种子字段并复用 LMEvalConfig 默认值。"""
    document = json.loads(args.eval_config.read_text(encoding="utf-8"))
    entries = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("评测配置必须包含非空 datasets 数组")
    configs = {}
    for entry in entries:
        if not isinstance(entry, dict) or "task" not in entry or "metrics" not in entry:
            raise ValueError("每个评测项必须包含 task 和 metrics")
        if (
            not isinstance(entry["task"], str)
            or not entry["task"].strip()
            or entry["task"] != entry["task"].strip()
        ):
            raise ValueError("task 必须为非空且无首尾空白的名称")
        values = dict(entry)
        metrics = values.pop("metrics")
        explicit = values.pop("metric_directions", None)
        if (
            not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(m, str) or not m.strip() for m in metrics)
        ):
            raise ValueError("metrics 必须为非空指标数组")
        metrics = [m.strip().lower() for m in metrics]
        if len(set(metrics)) != len(metrics):
            raise ValueError("metrics 不能重复")
        if explicit is None:
            resolved = resolve_metric_directions(metrics)
        else:
            if (
                not isinstance(explicit, dict)
                or set(explicit) != set(metrics)
                or any(v not in ("higher", "lower") for v in explicit.values())
            ):
                raise ValueError(
                    "metric_directions 必须准确覆盖 metrics 且方向为 higher/lower"
                )
            resolved = {m: explicit[m] for m in metrics}
        if "seed" in values:
            if "evaluation_seed" in values:
                raise ValueError("seed 和 evaluation_seed 不能同时指定")
            values["evaluation_seed"] = values.pop("seed")
        config = LMEvalConfig(**values)
        if not config.task.strip() or config.task in configs:
            raise ValueError("评测 task 必须非空且不能重复")
        configs[config.task] = EvaluationTaskConfig(config, resolved)
    return configs


def apply_evaluation_overrides(
    configs: Mapping[str, EvaluationTaskConfig],
    args: argparse.Namespace,
) -> dict[str, EvaluationTaskConfig]:
    """对已经统一的任务配置最后应用 CLI 题数及 batch size 覆盖。"""
    overrides = {}
    if args.eval_limit is not None:
        overrides["limit"] = args.eval_limit
    if args.eval_batch_size is not None:
        overrides["batch_size"] = args.eval_batch_size
    return {
        task: replace(config, evaluation=replace(config.evaluation, **overrides))
        for task, config in configs.items()
    }


def plotting_module() -> Any:
    """在模型加载前检查可选 Matplotlib 依赖，使用不需要显示器的 Agg 后端。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_scores(path: Path, summary: Mapping[str, Any]) -> None:
    """每个任务指标独立一张子图，标注原始得分及方向，不混用数值轴。"""
    plt = plotting_module()
    items = [
        (task, metric, values)
        for task, values_by_metric in summary["comparison"].items()
        for metric, values in values_by_metric.items()
    ]
    if not items:
        return
    fig, axes = plt.subplots(
        len(items), 1, figsize=(8, max(3, 2.8 * len(items))), squeeze=False
    )
    try:
        for ax, (task, metric, values) in zip(axes[:, 0], items, strict=True):
            bars = ax.bar(
                ["Baseline", "Compressed"],
                [values["baseline"], values["compressed"]],
                color=["#8ca49b", "#126a5b"],
            )
            ax.bar_label(
                bars,
                labels=[f"{values[key]:.6g}" for key in ("baseline", "compressed")],
                padding=5,
            )
            ax.set_title(f"{task} / {metric} ({values['direction']} is better)")
            ax.set_ylabel("Score (original units)")
            ax.margins(y=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """向 path 保存标准 JSON，拒绝非有限数值。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    """将实测参数收益和前后任务指标写成 Markdown，注明分数原单位及跳过状态。"""
    cm = summary["compression"]
    lines = [
        "# 联合压缩报告",
        "",
        f"- 模型：`{summary['model']}`",
        f"- 目标矩阵数：{cm['compressed_layers']}",
        f"- 参数量：{cm['dense_parameters']:,} → {cm['compressed_parameters']:,}",
        f"- 实测参数节省：{100 * summary['parameter_saving_fraction']:.4f}%",
        f"- 模型参数压缩比：{cm['compression_ratio']:.6f}",
        f"- 联合分解耗时：{summary['compression_seconds']:.3f} 秒",
        "",
    ]
    if summary["evaluation_status"] == "skipped":
        lines.append("本次跳过评测，联合压缩得分尚未评测。")
    else:
        lines += [
            "| 任务 | 指标 | 题数 | 原模型 | 联合压缩 | 退化（指标原单位，正值为变差） |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for task, values in summary["comparison"].items():
            for metric, value in values.items():
                lines.append(
                    f"| {task} | {metric} | {value['evaluated_examples']} | {value['baseline']:.6f} | {value['compressed']:.6f} | {value['degradation']:.6f} |"
                )
    lines += [
        "",
        "参数节省不代表推理加速。本次未微调；artifact 需与原始模型和执行后端配合使用。",
        "评测配置见 experiment_config.json；本次 baseline 重新评测，不直接使用单矩阵分析的历史分数。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """编排 JSON 驱动的压缩实验；dry-run 不加载模型，输出目录禁止覆盖。"""
    args = parse_args(argv)
    source_bytes = args.selection_json.read_bytes()
    document = json.loads(source_bytes)
    if (
        not isinstance(document, dict)
        or document.get("kind") != "qcomp_compression_selection"
    ):
        raise ValueError("请使用最新版选层页面导出的 JSON")
    # 昂贵模型加载前检查执行结构；加载后公共入口再检查实际目标维度。
    parsed = compression_plan_from_dict(document["compression_plan"])
    configs = evaluation_configs(document, args)
    compression_config = CompressionExecutionConfig(
        decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider,
        decomposition_dtype=DTYPES[args.decomposition_dtype],
    )
    runtime = load_runtime_config(args.runtime_config)
    model_name = (
        args.model
        or document.get("model", {}).get("name_or_path")
        or runtime.model_name_or_path
    )
    model_config = ModelLoadConfig(
        model_name_or_path=model_name,
        device=args.device,
        dtype=DTYPES[args.model_dtype],
        trust_remote_code=args.trust_remote_code,
    )
    output = args.output
    if output is None:
        slug = (
            re.sub(r"[^A-Za-z0-9._-]+", "-", Path(model_name).name).strip(".-")
            or "model"
        )
        output = (
            PROJECT
            / "artifacts/compression"
            / slug
            / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
        )
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"产物目录必须不存在：{output}")
    experiment = dict(
        model=model_name,
        selection_json=str(args.selection_json.resolve()),
        selection_sha256=hashlib.sha256(source_bytes).hexdigest(),
        output=str(output),
        device=args.device,
        model_dtype=args.model_dtype,
        decomposition_dtype=args.decomposition_dtype,
        decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider,
        trust_remote_code=args.trust_remote_code,
        decomposition_seed=args.decomposition_seed,
        offline=runtime.offline,
        runtime_config=args.runtime_config,
        target_count=len(parsed.targets),
        evaluation_configs={
            task: asdict(config.evaluation) for task, config in configs.items()
        },
        selected_metrics={
            task: dict(config.metric_directions) for task, config in configs.items()
        },
        skip_eval=args.skip_eval,
        eval_config=str(args.eval_config.resolve()) if args.eval_config else None,
        no_plot=args.no_plot,
    )
    if args.dry_run:
        print(json.dumps(experiment, ensure_ascii=False, indent=2))
        print("配置检查通过；尚未加载模型，目标类型与维度将在实际运行时校验。")
        return
    if not args.skip_eval and not args.no_plot:
        plotting_module()
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection.json").write_bytes(source_bytes)
    write_json(output / "experiment_config.json", experiment)
    log_path = output / "events.jsonl"
    with capture_console(output / "console.log"):
        log_event(log_path, "experiment_started", **experiment)
        artifacts = []
        phase = "loading"
        try:
            torch.manual_seed(args.decomposition_seed)
            print(f"产物目录：{output}\n加载模型：{model_name}", flush=True)
            resources = load_causal_lm(
                model_config, runtime_config_path=args.runtime_config
            )
            model = resources.model
            plan = load_compression_plan(output / "selection.json", model=model)
            plans = {"selected": plan}
            decomposition, execution = compression_config.build_backends(plans)
            evaluators = {
                task: LMEvalEvaluator(
                    resources.tokenizer,
                    config.evaluation,
                    runtime_config_path=args.runtime_config,
                )
                for task, config in configs.items()
            }

            def on_evaluation(
                plan_name: str | None, task: str, timed: TimedEvaluation
            ) -> None:
                """逐任务追加日志；耗时、指标和样本范围由工作流负责。"""
                stage = "baseline" if plan_name is None else "compressed"
                record = asdict(timed.evaluation)
                log_event(
                    log_path,
                    "evaluation_completed",
                    stage=stage,
                    task=task,
                    metrics=record["metrics"],
                    evaluated_examples=record["evaluated_examples"],
                    total_examples=record["total_examples"],
                    seconds=timed.seconds,
                )
                print(f"[{stage}] {task}: {record['metrics']}", flush=True)
                if plan_name is None and task == next(reversed(configs)):
                    # baseline 使用任务自己的种子，分解恢复独立的执行种子。
                    torch.manual_seed(args.decomposition_seed)

            def on_compressed(name, tensors, model_metrics, seconds) -> None:
                """在任务评测前保存逐矩阵 artifact，不保留张量引用。"""
                for index, (module_path, artifact) in enumerate(tensors.items()):
                    relative = f"decompositions/{index:04d}.pt"
                    save_artifact(output / relative, artifact)
                    artifacts.append(
                        dict(
                            module_path=module_path,
                            representation=artifact.representation,
                            path=relative,
                        )
                    )
                write_json(
                    output / "artifacts.json",
                    dict(
                        model=model_name,
                        execution_provider=args.execution_provider,
                        artifacts=artifacts,
                    ),
                )
                log_event(
                    log_path,
                    "compression_completed",
                    **asdict(model_metrics),
                    seconds=seconds,
                )

            def on_plan_result(completed: CompressionPlanEvaluation) -> None:
                """完整方案在恢复模型后记录完成事件。"""
                log_event(log_path, "plan_completed", name=completed.name)

            phase = "evaluation"
            torch.manual_seed(args.decomposition_seed)
            measured = evaluate_compression_plans(
                model,
                plans,
                evaluators=evaluators,
                metric_directions={
                    task: config.metric_directions for task, config in configs.items()
                },
                decomposition_backends=decomposition,
                execution_backends=execution,
                decomposition_dtype=compression_config.decomposition_dtype,
                on_evaluation=on_evaluation,
                on_compressed=on_compressed,
                on_plan_result=on_plan_result,
            )
            completed = measured.plan_results[0]
            cm = asdict(completed.model_compression)
            evaluations = {
                stage: {
                    task: dict(
                        metrics=dict(timed.evaluation.metrics),
                        evaluated_examples=timed.evaluation.evaluated_examples,
                        total_examples=timed.evaluation.total_examples,
                        seconds=timed.seconds,
                    )
                    for task, timed in records.items()
                }
                for stage, records in (
                    ("baseline", measured.baseline),
                    ("compressed", completed.evaluations),
                )
            }
            comparison = {
                task: {
                    metric: dict(
                        direction=direction,
                        baseline=measured.baseline[task].evaluation.metrics[metric],
                        compressed=completed.evaluations[task].evaluation.metrics[
                            metric
                        ],
                        degradation=completed.metric_degradations[task][metric],
                        evaluated_examples=measured.baseline[
                            task
                        ].evaluation.evaluated_examples,
                    )
                    for metric, direction in task_config.metric_directions.items()
                }
                for task, task_config in configs.items()
            }
            summary = dict(
                model=model_name,
                compression=cm,
                compression_seconds=completed.compression_seconds,
                parameter_saving_fraction=1 - 1 / cm["compression_ratio"],
                evaluation_status="skipped" if args.skip_eval else "completed",
                evaluations=evaluations,
                comparison=comparison,
                artifacts=artifacts,
            )
            phase = "output"
            write_json(output / "summary.json", summary)
            write_report(output / "report.md", summary)
            if not args.skip_eval and not args.no_plot:
                phase = "plotting"
                plot_scores(output / "scores.png", summary)
            log_event(
                log_path,
                "experiment_completed",
                summary="summary.json",
                report="report.md",
            )
            print(f"完成：{output / 'report.md'}", flush=True)
        except Exception as error:
            log_event(
                log_path,
                "experiment_failed",
                stage=phase,
                error_type=type(error).__name__,
                message=str(error),
            )
            raise


if __name__ == "__main__":
    main()
