"""从局部输出 NMSE 结果生成总体和逐来源热力图，并更新同目录报告。

本模块只读取已有实验统计，不加载模型或执行分解；入口脚本负责调用和命令行参数。
主要内容：
- ``plot_results``：按目标保留率生成共享色标的热力图，先总体、再逐来源。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


def plot_results(
    results_path: Path, modules: Sequence[str], vmax: float | None = None
) -> list[Path]:
    """读取结果 JSON，按来源和目标保留率绘图并幂等更新 report.md。

    参数：
        results_path: 局部 NMSE JSON 路径。
        modules: 热力图纵轴的固定投影顺序。
        vmax: 全部图片共享的色标上限；None 使用全部有效测量的最大值。

    返回：
        按总体、来源和保留率顺序排列的 PNG 路径。
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    if vmax is not None and (not math.isfinite(vmax) or vmax <= 0):
        raise ValueError("vmax must be finite and positive")
    results_path = Path(results_path)
    document = json.loads(results_path.read_text(encoding="utf-8"))
    experiment = document["experiment"]
    ratios = [float(value) for value in experiment["retention_ratios"]]
    sources = [entry["name"] for entry in experiment["calibration"]["datasets"]]
    streaming = experiment.get('protocol') == 'streaming-task-inputs-v1'
    if (not ratios and not streaming) or any(not 0 < value < 1 for value in ratios):
        raise ValueError("retention ratios must be between zero and one")
    labels = [f'rho:{value:g}' for value in ratios]
    if streaming:
        labels += list(dict.fromkeys(label for c in document['cases'] for label in c['labels'] if label.startswith('history:')))
    ratio_keys = labels if streaming else [round(value, 6) for value in ratios]
    if len(set(ratio_keys)) != len(ratio_keys) or len(set(sources)) != len(sources):
        raise ValueError("duplicate ratios or sources in heatmap configuration")
    cases = document["cases"]
    blocks = sorted({record["block"] for record in [*document["dense"], *cases]})
    if not blocks:
        raise ValueError("no blocks available for heatmaps")
    indexed = {}
    for record in cases:
        keys = record['labels'] if streaming else [round(record['requested_retention_ratio'], 6)]
        for group in keys:
            key = (record['block'], record['projection'], group)
            if key in indexed or key[1] not in modules or key[2] not in ratio_keys:
                raise ValueError(f'invalid or duplicate heatmap candidate: {key}')
            if record['status'] not in ('ok', 'failed', 'partial'):
                raise ValueError('unknown candidate status')
            indexed[key] = record

    panels = []
    display_sources = sources if streaming and len(sources) == 1 else [None, *sources]
    groups = ratio_keys if streaming else ratios
    for source in display_sources:
        for ratio, ratio_key in zip(groups, ratio_keys, strict=True):
            values = np.full((len(modules), len(blocks)), np.nan)
            failed = np.zeros_like(values, dtype=bool)
            tokens = set()
            actual_ratios = []
            for col, block in enumerate(blocks):
                for row, module in enumerate(modules):
                    record = indexed.get((block, module, ratio_key))
                    if record is None:
                        continue
                    if record["status"] == "failed":
                        failed[row, col] = True
                        continue
                    if source is not None and source not in record['sources']:
                        continue
                    stats = record if source is None else record['sources'][source]
                    if stats.get('output_nmse') is None:
                        continue
                    value = float(stats["output_nmse"])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("NMSE must be finite and non-negative")
                    count = stats["valid_tokens"]
                    if type(count) is not int or count <= 0:
                        raise ValueError("valid_tokens must be a positive integer")
                    values[row, col] = value
                    tokens.add(count)
                    actual_ratios.append(float(record["actual_retention_ratio"]))
            panels.append((source, ratio, values, failed, tokens, actual_ratios))
    maximum = max((float(np.nanmax(p[2])) for p in panels if np.isfinite(p[2]).any()), default=0.0)
    limit = vmax if vmax is not None else maximum or 1.0
    images = []
    sections = []
    for source, ratio, values, failed, tokens, actual_ratios in panels:
        label = "Overall" if source is None else source
        ratio_label = str(ratio) if streaming else f"rho-{ratio:g}"
        token_text = ", ".join(f"{count:,}" for count in sorted(tokens)) or "N/A"
        actual_text = (
            f"{min(actual_ratios):.4f}-{max(actual_ratios):.4f}" if actual_ratios else "N/A"
        )
        annotation = f"Valid calibration tokens / candidate: {token_text} | Actual rho: {actual_text}"
        fig = Figure(figsize=(max(9, len(blocks) * 0.36), 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        cmap = colormaps["viridis"].with_extremes(bad="#dddddd")
        im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0, vmax=limit)
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_yticks(range(len(modules)), modules)
        ax.set_xlabel("Transformer block")
        ax.set_ylabel("Module")
        ax.set_title(f"Qwen3 {label} output NMSE | Candidate: {ratio_label}\n{annotation}")
        fig.colorbar(im, ax=ax, label=f"Output NMSE (lower is better), scale [0, {limit:.4g}]")
        for row, col in np.argwhere(np.isfinite(values) & (values > limit)):
            ax.text(col, row, f"{values[row, col]:.3g}", color="black", ha="center", va="center", fontsize=7)
        for row, col in np.argwhere(failed):
            ax.text(col, row, "x", color="black", ha="center", va="center", fontsize=8)
        legend = []
        if (np.isnan(values) & ~failed).any():
            legend.append(Patch(color="#dddddd", label="Not evaluated / deduplicated"))
        if failed.any():
            legend.append(Patch(color="#dddddd", label="x: Failed"))
        if legend:
            fig.legend(handles=legend, loc="outside lower center", ncol=2)
        # 编码来源名称中的路径字符，前缀区分总体与同名数据来源。
        slug = "overall" if source is None else f"source-{quote(source, safe='')}"
        destination = results_path.with_name(f"local-output-nmse-{slug}-{ratio_label.replace(chr(58), chr(45))}-heatmap.png")
        if streaming:
            destination = results_path.with_name(f"qwen3-nmse-{slug}-{experiment['run_id']}-{ratio_label.replace(':', '-')}-heatmap.png")
        fig.savefig(destination, dpi=180, metadata={"Description": annotation})
        fig.clear()
        images.append(destination)
        if ratio == groups[0]:
            sections.extend([f"### {'总体' if source is None else source}", ""])
        sections.extend([f"![{label}, {ratio_label}]({quote(destination.name)})", ""])
    marker = "<!-- qcomp-local-output-nmse-heatmaps -->"
    report_path = results_path.with_name("report.md")
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Qwen3 local-output NMSE\n"
    introduction = (
        f"\n\n{marker}\n\n## 局部输出 NMSE 热力图\n\n"
        "按目标参数保留率分图，实际比例范围见标题。总体使用累计分子除以累计分母，"
        "各图共用色标；NMSE 越小越好，可以超过 1。灰格为缺失或 rank 去重项，"
        "灰格中的 x 表示失败；超过色标上限的格子标注实际值。dense 为零，不单独绘图。\n\n"
    )
    report_path.write_text(report.split(marker)[0].rstrip() + introduction + "\n".join(sections), encoding="utf-8")
    return images
