"""逐层执行 Qwen3 MPO 压缩与可选 lm-eval task 的敏感性分析。

本实验脚本提供 Qwen3 的层选择、MPO 结构配置 与分模块键维配置，通过上层 sensitivity 入口完成
模型加载、逐层 case 组装、临时压缩、评测、恢复和输出。
模型结构与 MPO modes 留在脚本中；默认评测 MMLU，task、指标和运行范围可以由命令行
覆盖。脚本直接调用通用 workflow，不经过 qcomp CLI 包。

主要内容：
- ``is_target_linear``：定义实验目标层的选择规则。
- ``qwen3_modes``、``make_qwen3_mpo_spec``：定义 Qwen3 的 MPO 结构。
- ``MODULE_RANKS``：定义不同 Linear 类型使用的 MPO 键维。
- ``parse_args``、``main``：解析实验参数、自动保存终端日志并调用通用 workflow。
- ``plot_heatmaps``、``plot_log``：生成模块与 Transformer 块热力图并支持已有日志补图。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from qcomp import (
    CompressionTarget,
    MPOSpec,
    ModelLoadConfig,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
)
from qcomp.logging import capture_console
from qcomp.evaluation import LMEvalConfig, lm_eval_dataset_size
from qcomp.workflows import sensitivity_case_record


def is_target_linear(path: str, linear: nn.Linear) -> bool:
    """判断一个 Qwen3 Linear 是否属于本次实验的候选集合。

    修改这个函数即可按模块路径、层类型或特征维度配置目标层；后续逻辑会自动为每个
    返回 ``True`` 的层创建独立 sensitivity case。

    参数：
        path: Linear 相对于模型根节点的模块路径。
        linear: 对应的稠密 Linear。

    返回：
        当前默认排除输出词表层，其他 Linear 全部参与。
    """

    del linear
    return path != "lm_head"


def qwen3_modes(size: int) -> tuple[int, int, int]:
    """返回 Qwen3-8B 已知特征维度对应的三核 MPO modes。

    参数：
        size: Linear 的输入或输出特征数。

    返回：
        乘积等于特征数的三个 mode。

    异常：
        ValueError: 特征数不属于当前 Qwen3-8B 配置时抛出。
    """

    modes = {
        1024: (8, 8, 16),
        2048: (8, 16, 16),
        4096: (16, 16, 16),
        6144: (16, 16, 24),
        12288: (16, 16, 48),
        151936: (8, 16, 1187),
    }
    if size not in modes:
        raise ValueError(f"no Qwen3 MPO modes configured for feature size {size}")
    return modes[size]


MODULE_RANKS: dict[str, int] = {
    # 当前配置按 Linear 权重形状调整 bond dimension，使不同模块的压缩率保持在
    # 约 7～8 倍，而不是对全部模块统一使用相同 rank：
    # - q_proj / o_proj：4096 × 4096，使用 rank 96；
    # - k_proj / v_proj：1024 × 4096，使用 rank 64；
    # - gate_proj / up_proj / down_proj：4096 与 12288 之间映射，使用 rank 160。
    #
    # 命令行显式传入 --rank 时，该统一 rank 会覆盖此默认分模块配置，用于运行
    # uniform-rank 对照实验。
    "q_proj": 96,
    "k_proj": 64,
    "v_proj": 64,
    "o_proj": 96,
    "gate_proj": 160,
    "up_proj": 160,
    "down_proj": 160,
}

def qwen3_full_ranks(
    out_modes: tuple[int, int, int],
    in_modes: tuple[int, int, int],
) -> tuple[int, int]:
    """计算三核 MPO 在不截断条件下允许的两条最大内部键维。

    参数：
        out_modes: Linear 输出维度对应的三个 MPO modes。
        in_modes: Linear 输入维度对应的三个 MPO modes。

    返回：
        两条内部 bond 的 full ranks ``(r1, r2)``。
    """

    physical_dims = tuple(
        out_mode * in_mode
        for out_mode, in_mode in zip(out_modes, in_modes)
    )
    n1, n2, n3 = physical_dims

    return (
        min(n1, n2 * n3),
        min(n1 * n2, n3),
    )

def make_qwen3_mpo_spec(
    linear: nn.Linear,
    ranks: tuple[int, int],
) -> MPOSpec:
    """为一个 Qwen3 Linear 创建三核 MPO spec。

    参数：
        linear: 待压缩的 Qwen3 Linear。
        ranks: 两条内部 MPO bonds 的键维 ``(r1, r2)``。

    返回：
        与 Linear 输入输出维度匹配的 MPO spec。

    异常：
        ValueError: 任一内部 rank 超出当前 MPO modes 的合法范围时抛出。
    """

    out_modes = qwen3_modes(linear.out_features)
    in_modes = qwen3_modes(linear.in_features)
    full_ranks = qwen3_full_ranks(out_modes, in_modes)

    if any(
        rank < 1 or rank > full_rank
        for rank, full_rank in zip(ranks, full_ranks)
    ):
        raise ValueError(
            f"MPO ranks {ranks} exceed valid full ranks {full_ranks} "
            f"for Linear({linear.in_features}, {linear.out_features})"
        )

    return MPOSpec(
        out_modes=out_modes,
        in_modes=in_modes,
        ranks=(1, *ranks, 1),
    )


def parse_limit(value: str) -> int | float | None:
    """解析 lm-eval 样本数、样本比例或无限制标记。

    参数：
        value: 正整数、(0, 1) 内的小数或 ``none``。

    返回：
        适用于 ``LMEvalConfig`` 的限制值。
    """

    if value.lower() == "none":
        return None
    return float(value) if "." in value else int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析模型、backend、lm-eval task、层范围和输出配置。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。

    返回：
        已补全默认 MMLU 指标的参数命名空间。
    """

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="逐层运行 Qwen3-8B MPO 的 lm-eval 敏感性分析。"
    )
    parser.add_argument(
        "--runtime-config",
        help="项目运行配置；相对路径按项目根目录解析。",
    )
    parser.add_argument(
        "--model",
        help="临时覆盖 runtime.toml 中的模型名称或本地路径。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )

    rank_group = parser.add_mutually_exclusive_group()

    rank_group.add_argument(
        "--rank",
        type=int,
        default=96,
        help=(
            "统一覆盖所有目标 Linear 的两条 MPO 内部键维；"
            "省略时使用默认分模块键维。"
        ),
    )

    rank_group.add_argument(
        "--full-rank",
        action="store_true",
        help="使用当前 MPO modes 的 full ranks,不进行 bond-rank 截断。",
    )

    parser.add_argument("--decomposition-provider", default="tensorly")
    parser.add_argument("--execution-provider", default="tensorly")
    parser.add_argument(
        "--start-layer-index", type=int, default=0,
        help="筛选后的 Linear 模块起始索引（从 0 开始，包含）。",
    )
    parser.add_argument(
        "--max-layers", type=int,
        help="最多分析的 Linear 模块数量；不是 Transformer block 数量。",
    )
    parser.add_argument("--task", default="mmlu")
    parser.add_argument(
        "--metric",
        action="append",
        help="参与退化比较的指标名称；可以重复传入。",
    )
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=parse_limit, default=1)
    parser.add_argument(
        "--sample-start-index", type=int, default=0,
        help="评测题目起始索引（从 0 开始，仅支持单 task）；limit 为从此处取的题数。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output")
    parser.add_argument("--log")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--plot-only", type=Path, help="仅从指定 JSONL 日志补图，不加载模型。"
    )
    parser.add_argument(
        "--heatmap-max",
        type=float,
        default=10.0,
        help="热力图色标上限；准确率类指标单位为百分点，默认 10。",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.heatmap_max) or args.heatmap_max <= 0:
        parser.error("--heatmap-max must be finite and positive")
    args.task = args.task.strip()
    if not args.task:
        parser.error("--task must not be empty")
    if args.metric is None:
        if args.task != "mmlu" and args.plot_only is None:
            parser.error("--metric is required when --task is not mmlu")
        args.metric = ["acc"]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """执行逐层压缩、指定 lm-eval task 评测、恢复和报告输出。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。
    """

    args = parse_args(argv)
    if args.plot_only is not None:
        plot_log(args.plot_only, args.heatmap_max)
        return
    
    # 根据命令行参数选择本次实验使用的 MPO rank 策略。
    if args.full_rank:
        rank_strategy = "full-rank"
    elif args.rank is not None:
        rank_strategy = f"rank-{args.rank}"
    else:
        rank_strategy = "module-ranks"
        
    config = SensitivityExperimentConfig(
        name=f"qwen3-{args.task}-mpo-{rank_strategy}",
        model=ModelLoadConfig(
            model_name_or_path=args.model,
            device=args.device,
            dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[
                args.model_dtype
            ],
            trust_remote_code=args.trust_remote_code,
        ),
        evaluation=LMEvalConfig(
            task=args.task,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            sample_start_index=args.sample_start_index,
            seed=args.seed,
            apply_chat_template=args.apply_chat_template,
        ),
        metrics=tuple(args.metric),
        runtime_config=args.runtime_config,
        decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider,
        start_layer_index=args.start_layer_index,
        max_layers=args.max_layers,
        artifact_root=args.artifact_root,
        output=args.output,
        log=args.log,
    )


    def make_target(path: str, linear: nn.Linear) -> CompressionTarget:
        """根据实验策略为 Qwen3 Linear 创建 MPO 压缩目标。

        参数：
            path: Linear 相对于模型根节点的模块路径。
            linear: 待压缩的稠密 Linear。

        返回：
            使用 full rank、统一 rank 或默认分模块 rank 构造的压缩目标。
        """

        out_modes = qwen3_modes(linear.out_features)
        in_modes = qwen3_modes(linear.in_features)

        if args.full_rank:
            ranks = qwen3_full_ranks(out_modes, in_modes)
        elif args.rank is not None:
            ranks = (args.rank, args.rank)
        else:
            module_name = path.rsplit(".", 1)[-1]
            rank = MODULE_RANKS[module_name]
            ranks = (rank, rank)

        return CompressionTarget(
            module_path=path,
            representation="mpo",
            spec=make_qwen3_mpo_spec(linear, ranks),
        )

    if config.output is None:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y%m%dT%H%M%S"
        )
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", config.name).strip("-._")
        directory = Path(config.artifact_root) / "sensitivity" / slug / timestamp
        config = replace(config, output=directory / "report.md")
    report_path = Path(config.output)
    if config.log is None:
        config = replace(config, log=report_path.parent / "events.jsonl")
    with capture_console(report_path.parent / "console.log"):
        print(f"Experiment directory: {report_path.parent.resolve()}", flush=True)
        result = run_sensitivity_experiment(
            config, select_linear=is_target_linear, make_target=make_target
        )

        if result.report_path is None:
            raise ValueError("experiment did not return a report path")
        plot_heatmaps(
            [sensitivity_case_record(case) for case in result.case_results],
            task=args.task,
            metrics=config.metrics,
            report_path=result.report_path,
            vmax=args.heatmap_max,
            evaluated_examples=result.baseline_evaluation.evaluated_examples,
            total_examples=result.baseline_evaluation.total_examples,
        )


MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def heatmap_values(
    records: Sequence[Mapping[str, Any]],
    metric: str,
) -> tuple[Any, list[int], str]:
    """将单层记录转换为七行矩阵，准确率类退化量乘以 100 转为百分点。

    参数：
        records: 公共 sensitivity_case_record 格式的记录。
        metric: 日志中的退化指标名。

    返回：
        缺失格为 NaN 的矩阵、连续块号和单位。

    异常：
        ValueError: 记录为空、非单模块实验、坐标重复或指标非有限数。
    """
    import numpy as np

    points = {}
    unit = (
        "pp"
        if metric in {"acc", "acc_norm", "f1"} or metric.startswith("exact_match")
        else "raw units"
    )
    for record in records:
        paths = list(record["layers"])
        if len(paths) != 1:
            raise ValueError("heatmaps require single-module cases")
        match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)", paths[0])
        if match is None or match[3] not in MODULES:
            raise ValueError(f"unsupported Qwen3 module: {paths[0]}")
        block, module = int(match[1]), match[3]
        key = (MODULES.index(module), block)
        value = float(record["degradations"][metric])
        if key in points or not math.isfinite(value):
            raise ValueError(
                f"duplicate coordinate or non-finite degradation: {paths[0]}"
            )
        points[key] = value * (100 if unit == "pp" else 1)
    if not points:
        raise ValueError("no completed cases to plot")
    blocks = list(range(min(b for _, b in points), max(b for _, b in points) + 1))
    values = np.full((len(MODULES), len(blocks)), np.nan)
    for (row, block), value in points.items():
        values[row, block - blocks[0]] = value
    return values, blocks, unit


def plot_heatmaps(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    metrics: Sequence[str],
    report_path: Path,
    vmax: float = 10.0,
    evaluated_examples: int | None = None,
    total_examples: int | None = None,
) -> list[Path]:
    """在 report_path 旁保存各 metrics 的 PNG，并在报告末尾嵌入图片。

    参数：
        records: 单模块实验记录。
        task: 图标题中的任务名称。
        metrics: 每个指标生成一张图。
        report_path: 已有 Markdown 报告位置。
        vmax: 色标上限，越界格标出真实值；负值表示改善。
        evaluated_examples: 本次每轮实际测试条数；省略时读取报告。
        total_examples: 完整评测集总条数；省略时读取报告或本地 task 定义。

    返回：
        生成的图片路径列表。
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch

    if not math.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be finite and positive")
    report_path = Path(report_path)
    report = report_path.read_text(encoding="utf-8")
    if evaluated_examples is None:
        count = re.search(r"^- Evaluated examples: (\d+)$", report, re.MULTILINE)
        if count is None:
            raise ValueError("report is missing evaluated sample count")
        evaluated_examples = int(count[1])
    if total_examples is None:
        count = re.search(r"^- Total evaluation examples: (\d+)$", report, re.MULTILINE)
        total_examples = int(count[1]) if count else lm_eval_dataset_size(task)
    if not 0 < evaluated_examples <= total_examples:
        raise ValueError("sample counts must satisfy 0 < evaluated <= total")
    annotation = f"Test samples: {evaluated_examples:,} / {total_examples:,} (evaluation split)"
    if not re.search(r"^- Total evaluation examples:", report, re.MULTILINE):
        report = report.replace(
            f"- Evaluated examples: {evaluated_examples}",
            f"- Evaluated examples: {evaluated_examples}\n- Total evaluation examples: {total_examples}",
            1,
        )
    images = []
    for metric in metrics:
        values, blocks, unit = heatmap_values(records, metric)
        fig = Figure(figsize=(max(9, len(blocks) * 0.36), 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        cmap = colormaps["viridis"].with_extremes(bad="#dddddd", under="#24243e")
        im = ax.imshow(
            np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0, vmax=vmax
        )
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_yticks(range(len(MODULES)), MODULES)
        ax.set_xlabel("Transformer block")
        ax.set_ylabel("Module")
        ax.set_title(f"{task.upper()} {metric} degradation heatmap ({unit})\n{annotation}")
        fig.colorbar(im, ax=ax, label=f"Degradation ({unit}), clipped to [0, {vmax:g}]")
        for row, col in np.argwhere(
            np.isfinite(values) & ((values > vmax) | (values < 0))
        ):
            ax.text(
                col,
                row,
                (
                    f"{values[row, col]:.1g}"
                    if values[row, col] < 0
                    else f"{values[row, col]:.1f}"
                ),
                color="white" if values[row, col] < 0 else "black",
                ha="center",
                va="center",
                fontsize=6 if values[row, col] < 0 else 7,
            )
        legend = []
        if np.isnan(values).any():
            legend.append(Patch(color="#dddddd", label="Not evaluated"))
        if (values < 0).any():
            legend.append(Patch(color="#24243e", label="Improvement (< 0)"))
        if legend:
            fig.legend(handles=legend, loc="outside lower center", ncol=2)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", metric)
        destination = report_path.with_name(f"{report_path.stem}-{slug}-heatmap.png")
        fig.savefig(destination, dpi=180, metadata={"Description": annotation})
        images.append(destination)
        print(f"Heatmap: {destination.resolve()}")
    marker = "<!-- qcomp-sensitivity-heatmaps -->"
    report = report.split(marker)[0].rstrip()
    section = "\n\n" + marker + "\n\n## Sensitivity heatmaps\n\n"
    section += (
        "Positive values indicate degradation; negative values indicate improvement. "
    )
    section += "Accuracy and exact-match drops are in percentage points (pp). "
    section += "Out-of-range cells show their actual values; gray cells were not evaluated.\n\n"
    section += "\n\n".join(
        f"![{metric} heatmap]({quote(path.name)})"
        for metric, path in zip(metrics, images)
    )
    report_path.write_text(report + section + "\n", encoding="utf-8")
    return images


def plot_log(log_path: Path, vmax: float = 10.0) -> list[Path]:
    """读取 log_path 中单次已完成实验的 JSONL，按记录中的任务和指标补图。

    参数：
        log_path: 包含开始、逐层结果和完成事件的日志。
        vmax: 热力图色标上限。

    异常：
        ValueError: 日志混合多次运行、缺少完成事件或运行标识不一致。
    """
    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    starts = [e["fields"] for e in events if e["event"] == "experiment_started"]
    ends = [e["fields"] for e in events if e["event"] == "experiment_completed"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("plot-only requires exactly one completed experiment")
    start = starts[0]
    records = [e["fields"] for e in events if e["event"] == "case_completed"]
    if any(r.get("run_id") != start["run_id"] for r in [*records, ends[0]]):
        raise ValueError("mixed run IDs in log")
    if len(records) != start["layers"]:
        raise ValueError("completed case count differs from experiment configuration")
    return plot_heatmaps(
        records,
        task=start["task"],
        metrics=start["metrics"],
        report_path=Path(ends[0]["report"]),
        vmax=vmax,
    )


if __name__ == "__main__":
    main()
