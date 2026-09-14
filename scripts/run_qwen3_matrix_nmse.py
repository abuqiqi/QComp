"""按层计算 Qwen3 MPO 分解后参数矩阵的 NMSE。

本脚本只做参数矩阵重构误差统计，不进行 lm-eval。它复用已有 Qwen3 MPO 结构规则，
支持统一 rank、分模块 rank 和满秩三种策略，并输出 JSON 与默认明细 CSV/heatmap。

主要内容：
- ``is_target_linear``：复用灵敏度脚本风格的目标层过滤。
- ``make_qwen3_mpo_spec``：按策略生成 MPO spec。
- ``evaluate_weight_nmse``：对单层权重进行分解-重构并计算 NMSE。
- ``_extract_module_coordinates``：从模块路径提取层号与模块名。
- ``write_case_csv``：输出明细 CSV。
- ``plot_nmse_heatmap``：按 Transformer block 画 NMSE heatmap。
- ``main``：加载模型、遍历目标层并保存 JSON 与默认明细 CSV、heatmap。

使用说明（以下命令在项目根目录执行）:
    python scripts/run_qwen3_matrix_nmse.py --max-layers 8 --rank 96
    python scripts/run_qwen3_matrix_nmse.py --max-layers 8 --module-ranks
    python scripts/run_qwen3_matrix_nmse.py --max-layers 2 --full-rank --output /tmp/qwen3_matrix_nmse.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qcomp.evaluation import compute_nmse
if __package__:
    from .qwen3_mpo_config import MODULE_RANKS, qwen3_mpo_spec_dict, qwen3_modes
else:
    from qwen3_mpo_config import MODULE_RANKS, qwen3_mpo_spec_dict, qwen3_modes

from qcomp import (
    MPOSpec,
    ModelLoadConfig,
    get_backend,
    list_linears,
    load_causal_lm,
    load_runtime_config,
    reconstruct_mpo,
)

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LINEAR_PATH_RE = re.compile(
    r".*\.layers\.(?P<block>\d+)\.(?:self_attn|mlp)\.(?P<module>\w+)$"
)


def _model_output_paths(output: str) -> tuple[Path, Path, Path]:
    """从输出参数派生 JSON、CSV 与 heatmap 路径。

    参数：
        output: 命令行传入的输出路径。

    返回：
        ``(json_path, csv_path, heatmap_path)``。
    """
    output_path = Path(output)
    if output_path.suffix != ".json":
        output_path = output_path.with_suffix(".json")
    csv_path = output_path.with_suffix(".csv")
    heatmap_path = output_path.with_name(f"{output_path.stem}-nmse-heatmap.png")
    return output_path, csv_path, heatmap_path


def _model_slug(model_name: str | None, runtime_config: str | None = None) -> str:
    """从模型来源与 runtime 配置提取目录安全的模型标识。

    参数：
        model_name: 传入的 model 字段或覆盖值。
        runtime_config: runtime 文件路径。

    返回：
        用于实验目录命名的模型标识。

    异常：
        ValueError: model 名称为空且 runtime 配置缺失时抛出。
    """

    default = "qwen3"
    if model_name is None:
        runtime = load_runtime_config(runtime_config)
        model_name = runtime.model_name_or_path
    candidate = model_name.strip().rstrip("/")
    if not candidate:
        return default
    candidate = candidate.rsplit("/", 1)[-1]
    match = re.search(r"(qwen3-[0-9.]+)b", candidate, re.IGNORECASE)
    if match is not None:
        return f"{match.group(1).lower()}B"
    candidate_lower = candidate.lower()
    if candidate_lower.startswith("qwen3"):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", candidate_lower).strip("-._")
    return default


def is_target_linear(path: str, linear: nn.Linear) -> bool:
    """复用灵敏度脚本风格的目标 Linear 规则。

    参数：
        path: 线性层完整路径。
        linear: 线性层对象。

    返回：
        不是 ``lm_head`` 时返回 ``True``。
    """
    del linear
    return path != "lm_head"


def _extract_module_coordinates(path: str) -> tuple[int, str]:
    """从层路径提取 block 编号和模块名。

    参数：
        path: 目标线性层的路径，例如 ``model.layers.4.self_attn.q_proj``。

    返回：
        ``(block_index, module_name)``。

    异常：
        ValueError: 路径格式不匹配。
    """
    match = LINEAR_PATH_RE.fullmatch(path)
    if match is None:
        raise ValueError(f"不支持的层路径：{path}")
    block = int(match["block"])
    module = match["module"]
    if module not in MODULES:
        raise ValueError(f"不支持的模块类型：{module}")
    return block, module


def write_case_csv(records: list[dict[str, Any]], csv_path: Path) -> Path:
    """按层明细写入 CSV。

    参数：
        records: 单层 NMSE 记录列表。
        csv_path: CSV 输出路径。

    返回：
        写入完成的 CSV 路径。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "module",
        "status",
        "module_rank",
        "out_features",
        "in_features",
        "numel",
        "nmse",
        "mse",
        "target_power",
        "error",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})
    return csv_path


def plot_nmse_heatmap(records: list[dict[str, Any]], output: Path) -> Path:
    """将成功层结果绘制为 NMSE 热力图。

    参数：
        records: 按 ``status``、``nmse`` 与模块路径构成的明细。
        output: PNG 输出路径。

    返回：
        生成的 heatmap 路径。

    异常：
        ValueError: 无可绘制样本、坐标重复或 NMSE 非有限。
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points: dict[tuple[int, int], float] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        nmse = record.get("nmse")
        if not isinstance(nmse, float | int) or not math.isfinite(float(nmse)):
            raise ValueError(f"非有限 NMSE 值：{record['module']}")
        block, module = _extract_module_coordinates(record["module"])
        key = (MODULES.index(module), block)
        if key in points:
            raise ValueError(f"重复坐标：{record['module']}")
        points[key] = float(nmse)

    if not points:
        raise ValueError("No valid OK records to plot")

    block_values = sorted({block for _, block in points})
    values = np.full((len(MODULES), len(block_values)), np.nan, dtype=np.float64)
    for (row, block), value in points.items():
        values[row, block_values.index(block)] = value

    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1.0
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = Figure(figsize=(max(9, len(block_values) * 0.38), 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    cmap = colormaps["viridis"].with_extremes(bad="#dddddd", under="#24243e")
    im = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax * 1.1,
    )
    ax.set_xticks(range(len(block_values)), block_values)
    ax.set_yticks(range(len(MODULES)), MODULES)
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Module")
    ax.set_title("Qwen3 matrix NMSE heatmap")
    fig.colorbar(im, ax=ax, label="NMSE")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Description": "Qwen3 matrix NMSE heatmap"})
    return output


def make_qwen3_mpo_spec(linear: nn.Linear, rank: int | None) -> MPOSpec:
    """为一个 Qwen3 Linear 创建 MPO spec。

    参数：
        linear: 目标线性层。
        rank: 分解 rank；``None`` 表示满秩。

    返回：
        MPO spec。
    """
    if rank is None:
        return MPOSpec.full_rank(
            out_modes=qwen3_modes(linear.out_features),
            in_modes=qwen3_modes(linear.in_features),
        )
    return MPOSpec(**qwen3_mpo_spec_dict(linear.out_features, linear.in_features, rank))


def evaluate_weight_nmse(
    *,
    weight: torch.Tensor,
    decompose_dtype: torch.dtype,
    rank: int | None,
    backend,
) -> dict[str, float]:
    """按给定 rank 进行 MPO 分解并计算矩阵 NMSE。

    参数：
        weight: 输入权重。
        decompose_dtype: 分解使用的数据类型。
        rank: 分解 rank；``None`` 表示满秩。
        backend: MPO 后端实现。

    返回：
        包含 ``nmse``、``mse``、``target_power`` 的字典。
    """
    spec = make_qwen3_mpo_spec(
        nn.Linear(weight.shape[1], weight.shape[0], bias=False),
        rank,
    )
    artifact = backend.decompose(weight.to(dtype=decompose_dtype), spec)
    reconstructed = reconstruct_mpo(artifact).to(dtype=weight.dtype)
    metrics = compute_nmse(predictions=reconstructed, targets=weight, eps=1e-12)
    return {
        "nmse": float(metrics["nmse"].item()),
        "mse": float(metrics["mse"].item()),
        "target_power": float(metrics["target_power"].item()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析矩阵 NMSE 实验参数。

    参数：
        argv: 命令行参数列表。

    返回：
        解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="计算 Qwen3 逐层参数矩阵 MPO 重构 NMSE。",
    )
    parser.add_argument(
        "--runtime-config",
        help="项目运行配置；相对路径按项目根目录解析。",
    )
    parser.add_argument(
        "--model", help="临时覆盖 runtime.toml 中的模型名称或本地路径。"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--decompose-dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
    )
    ranks = parser.add_mutually_exclusive_group()
    ranks.add_argument("--rank", type=int, help="统一内部 rank；默认 96。")
    ranks.add_argument(
        "--module-ranks", action="store_true", help="使用分模块 rank。"
    )
    ranks.add_argument(
        "--full-rank", action="store_true", help="使用 MPO 满秩。"
    )
    parser.add_argument(
        "--decomposition-provider",
        default="tensorly",
        help="分解后端，默认 tensorly。",
    )
    parser.add_argument(
        "--start-layer-index",
        type=int,
        default=0,
        help="筛选后的 Linear 模块起始索引（从 0 开始，包含）。",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        help="最多测试的 Linear 模块数量；不是 Transformer block 数量。",
    )
    parser.add_argument(
        "--output",
        help=(
            "输出 JSON 路径；默认输出到"
            " artifacts/sensitivity/<模型>/matrix-nmse/<时间戳>.json。"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到单层失败直接退出，否则记录 error 后继续。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载远端模型时允许执行远端代码。",
    )

    args = parser.parse_args(argv)
    if args.rank is None and not args.module_ranks and not args.full_rank:
        args.rank = 96
    if args.rank is not None and args.rank <= 0:
        parser.error("--rank must be positive")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive")
    if args.start_layer_index < 0:
        parser.error("--start-layer-index must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """执行模型加载、目标层遍历与 NMSE 记录并写入结果文件。"""

    args = parse_args(argv)
    dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    decompose_dt_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    resources = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=args.model,
            device=args.device,
            dtype=dt_map[args.model_dtype],
            trust_remote_code=args.trust_remote_code,
        ),
        runtime_config_path=args.runtime_config,
    )
    model = resources.model

    backend = get_backend(args.decomposition_provider, "mpo")
    probe = backend.probe()
    if not probe.available:
        raise RuntimeError(f"backend unavailable: {probe.reason}")

    selected = 0
    rank_strategy = (
        "full-rank"
        if args.full_rank
        else f"rank-{args.rank}" if args.rank is not None else "module-ranks"
    )
    records: list[dict[str, Any]] = []
    all_nmse: list[float] = []

    for path, linear in list_linears(model):
        if not is_target_linear(path, linear):
            continue
        module_name = path.rsplit(".", 1)[-1]
        if module_name not in MODULES:
            continue

        if selected < args.start_layer_index:
            selected += 1
            continue
        if args.max_layers is not None and len(records) >= args.max_layers:
            break

        if args.module_ranks:
            rank = MODULE_RANKS.get(module_name)
            if rank is None:
                raise ValueError(f"缺少模块分 rank 配置：{module_name}")
        elif args.full_rank:
            rank = None
        else:
            rank = args.rank

        try:
            metrics = evaluate_weight_nmse(
                weight=linear.weight.detach(),
                decompose_dtype=decompose_dt_map[args.decompose_dtype],
                rank=rank,
                backend=backend,
            )
            all_nmse.append(metrics["nmse"])
            status = "ok"
        except Exception as exc:
            if args.fail_fast:
                raise
            status = "failed"
            metrics = {
                "nmse": None,
                "mse": None,
                "target_power": None,
                "error": str(exc),
            }

        records.append(
            {
                "module": path,
                "status": status,
                "module_rank": rank,
                "out_features": linear.out_features,
                "in_features": linear.in_features,
                "numel": linear.weight.numel(),
                **metrics,
            }
        )
        selected += 1

    successful = [x for x in records if x.get("status") == "ok"]
    summary = {
        "model": args.model or load_runtime_config(args.runtime_config).model_name_or_path,
        "decomposition_provider": args.decomposition_provider,
        "model_dtype": args.model_dtype,
        "decompose_dtype": args.decompose_dtype,
        "rank_strategy": rank_strategy,
        "rank": args.rank,
        "module_ranks": bool(args.module_ranks),
        "full_rank": bool(args.full_rank),
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "mean_nmse": float(sum(all_nmse) / len(all_nmse)) if all_nmse else None,
        "max_nmse": float(max(all_nmse)) if all_nmse else None,
        "min_nmse": float(min(all_nmse)) if all_nmse else None,
    }

    if args.output is None:
        runtime_model = args.model
        if runtime_model is None:
            runtime_model = load_runtime_config(args.runtime_config).model_name_or_path
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
        directory = Path("artifacts") / "sensitivity" / f"{_model_slug(runtime_model)}-matrix-nmse" / rank_strategy
        args.output = str(directory / f"{timestamp}.json")

    output, csv_output, heatmap_output = _model_output_paths(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_case_csv(records, csv_output)
    heatmap_path = plot_nmse_heatmap(records, output=heatmap_output)

    print(
        json.dumps(
            {"output": str(output), "csv": str(csv_output), "heatmap": str(heatmap_path), "summary": summary},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
