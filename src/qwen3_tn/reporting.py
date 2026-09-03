"""Human-readable weekly report updates for the CompactifAI-style run."""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .provenance import load_json

AUTO_START = "<!-- AUTO_RESULTS_START -->"
AUTO_END = "<!-- AUTO_RESULTS_END -->"


def _json_or_empty(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    return load_json(path)


def _fmt_number(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else "尚无"


def _fmt_float(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "尚无"


def _fmt_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "尚无"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours} 小时 {minutes} 分 {secs} 秒"


def _elapsed(state: Mapping[str, Any], metrics: Mapping[str, Any]) -> float | None:
    start = _display_start_time(state)
    end = state.get("end_time")
    if isinstance(start, str) and isinstance(end, str):
        try:
            return (
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
        except ValueError:
            pass
    value = metrics.get("training_seconds_this_run")
    return float(value) if isinstance(value, (int, float)) else None


def _display_start_time(state: Mapping[str, Any]) -> Any:
    if state.get("status") == "success":
        latest = state.get("last_resume_time")
        if isinstance(latest, str):
            return latest
    return state.get("start_time")


def _link(report_path: Path, target: str | Path | None, label: str) -> str:
    if target is None:
        return "尚无"
    path = Path(target)
    relative = os.path.relpath(path.resolve(), report_path.parent.resolve())
    return f"[{label}]({relative})"


def _error_summary(state: Mapping[str, Any]) -> str:
    explicit = state.get("error")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()[:500]
    log = state.get("log_path")
    if isinstance(log, str) and Path(log).is_file():
        lines = [
            line.strip()
            for line in Path(log)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
            if line.strip()
        ]
        if lines:
            return " / ".join(lines[-3:])[-500:]
    return "日志中没有可提取的错误摘要，请查看完整日志。"


def _loss_summary(
    state: Mapping[str, Any], metrics: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    values: list[float] = []
    log = state.get("log_path")
    if isinstance(log, str) and Path(log).is_file():
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        values = [
            float(match.group(1))
            for match in re.finditer(
                r"(?:^|\s)step=\d+\s+loss=([0-9]+(?:\.[0-9]+)?)", text
            )
        ]
    if values:
        return values[0], values[-1], min(values)
    return (
        metrics.get("first_training_loss"),
        metrics.get("last_training_loss"),
        metrics.get("min_training_loss"),
    )


def render_auto_results(
    report_path: str | Path,
    state: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
    decomposition: Mapping[str, Any] | None = None,
    verification: Mapping[str, Any] | None = None,
) -> str:
    """Render only the machine-controlled section; no raw JSON is embedded."""

    path = Path(report_path)
    run = dict(run or {})
    decomposition = dict(decomposition or {})
    verification = dict(verification or {})
    metrics = dict(run.get("training_metrics", {}))
    aggregate = dict(decomposition.get("aggregate", {}))
    status = str(state.get("status", "pending"))
    status_names = {
        "pending": "未开始",
        "running": "运行中",
        "success": "成功",
        "failure": "失败",
    }
    status_text = status_names.get(status, status)
    resumed = metrics.get("resumed_from") or state.get("resumed_from")
    attempt_count = int(state.get("attempt_count", 1))
    interrupted = max(0, attempt_count - 1)
    checkpoints = []
    artifact_root = state.get("artifact_root")
    if isinstance(artifact_root, str):
        checkpoints = sorted(Path(artifact_root).glob("checkpoint-*"))
    dense = aggregate.get("dense_parameters", 2_566_914_048)
    tt = aggregate.get(
        "tt_parameters", metrics.get("trainable_tt_parameters", 218_972_160)
    )
    ratio = aggregate.get("compression_ratio")
    if (
        not isinstance(ratio, (int, float))
        and isinstance(dense, (int, float))
        and isinstance(tt, (int, float))
        and tt
    ):
        ratio = dense / tt
    full_before = 8_190_735_360
    full_after = (
        full_before - int(dense) + int(tt)
        if isinstance(dense, (int, float)) and isinstance(tt, (int, float))
        else None
    )
    reduction = 100 * (full_before - full_after) / full_before if full_after else None
    steps = metrics.get("global_step")
    tokens = metrics.get("training_tokens_seen")
    if tokens is None and isinstance(steps, (int, float)):
        tokens = int(steps) * 4 * 512
    peak = metrics.get("peak_cuda_allocated_bytes")
    peak_gib = float(peak) / 1024**3 if isinstance(peak, (int, float)) else None
    final_index = (
        Path(str(artifact_root)) / "final" / "index.json" if artifact_root else None
    )
    last_checkpoint = checkpoints[-1] if checkpoints else None
    first_loss, last_loss, min_loss = _loss_summary(state, metrics)
    loss_change = (
        last_loss - first_loss
        if all(isinstance(value, (int, float)) for value in (first_loss, last_loss))
        else None
    )
    generation = verification.get("generated_text", "尚无")
    if isinstance(generation, str):
        generation = generation.replace("\n", " ").strip()[:400] or "（生成结果为空）"
    lines = [
        AUTO_START,
        "## 实验结果（脚本自动更新）",
        "",
        f"**当前状态：{status_text}。** 当前阶段：{state.get('stage', '尚未进入执行阶段')}。",
        "",
        "| 项目 | 实际结果 |",
        "|---|---|",
        f"| 开始 / 结束 | {_display_start_time(state) or '尚无'} / {state.get('end_time', '尚无')} |",
        f"| {'成功主流程耗时' if status == 'success' else '总耗时'} | {_fmt_duration(_elapsed(state, metrics))} |",
    ]
    if status != "success":
        lines.append(
            f"| 中断 / 续训 | 记录到 {interrupted} 次中断；"
            f"{'从 ' + str(resumed) + ' 恢复' if resumed else '本次未从 checkpoint 恢复'} |"
        )
    lines.extend(
        [
        f"| optimizer steps / 实际训练 tokens | {_fmt_number(steps)} / {_fmt_number(tokens)} |",
        f"| 观测首段 / 末段 / 最低 loss | {_fmt_float(first_loss)} / {_fmt_float(last_loss)} / {_fmt_float(min_loss)} |",
        f"| loss 变化（末段－首段） | {_fmt_float(loss_change)}；负数表示训练日志中的 loss 总体下降 |",
        f"| 峰值 GPU 显存 | {_fmt_float(peak_gib, 2)} GiB |",
        f"| 保留的 checkpoint | {len(checkpoints)} 个；最后一个：{_link(path, last_checkpoint, 'checkpoint')} |",
        f"| 最终 TT artifact | {_link(path, final_index, 'final/index.json')} |",
        f"| 目标矩阵参数量 | {_fmt_number(dense)} → {_fmt_number(tt)}，约 {_fmt_float(ratio, 2)}× |",
        f"| 完整模型参数量 | {_fmt_number(full_before)} → {_fmt_number(full_after)}，减少 {_fmt_float(reduction, 2)}% |",
        f"| 最终模型离线重载 | {'成功' if verification.get('success') is True else '尚未成功验证'} |",
        f"| 原始训练记录 | {_link(path, state.get('run_json'), 'run.json')} |",
        f"| 原始分解记录 / 日志 | {_link(path, state.get('decomposition_json'), 'decomposition result')} / {_link(path, state.get('log_path'), '日志')} |",
        "",
        "固定 prompt 的生成结果：",
        "",
        f"> {generation}",
        ]
    )
    if status == "failure":
        lines.extend(
            [
                "",
                f"失败阶段：{state.get('stage', '未知')}。错误摘要：{_error_summary(state)}",
                "",
                f"最后有效 checkpoint：{_link(path, last_checkpoint, 'checkpoint')}；完整日志：{_link(path, state.get('log_path'), '日志')}。",
            ]
        )
    elif status in {"pending", "running"}:
        lines.extend(
            [
                "",
                "这些位置显示“尚无”是正常的；训练完成或失败后脚本会再次更新本节。",
                "",
                f"运行日志：{_link(path, state.get('log_path'), '日志')}。",
            ]
        )
    lines.extend(["", AUTO_END])
    return "\n".join(lines)


def update_report(
    report_path: str | Path,
    state: Mapping[str, Any],
    *,
    run_json: str | Path | None = None,
    decomposition_json: str | Path | None = None,
    verification_json: str | Path | None = None,
) -> None:
    """Atomically replace the marked results section and preserve manual prose."""

    target = Path(report_path)
    content = target.read_text(encoding="utf-8")
    if content.count(AUTO_START) != 1 or content.count(AUTO_END) != 1:
        raise ValueError("report must contain exactly one controlled result section")
    before, remainder = content.split(AUTO_START, 1)
    _, after = remainder.split(AUTO_END, 1)
    section = render_auto_results(
        target,
        state,
        run=_json_or_empty(run_json),
        decomposition=_json_or_empty(decomposition_json),
        verification=_json_or_empty(verification_json),
    )
    updated = before.rstrip() + "\n\n" + section + after
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, target)
