"""逐层执行 Qwen3 MPO 压缩与可选 lm-eval task 的敏感性分析。

本实验脚本提供 Qwen3 的层选择与 MPO 结构配置，通过上层 sensitivity 入口完成
模型加载、逐层 case 组装、临时压缩、评测、恢复和输出。
模型结构与 MPO modes 留在脚本中；默认评测 MMLU，task、指标和运行范围可以由命令行
覆盖。脚本直接调用通用 workflow，不经过 qcomp CLI 包。

主要内容：
- ``is_target_linear``：定义实验目标层的选择规则。
- ``qwen3_modes``、``make_qwen3_mpo_spec``：定义 Qwen3 的 MPO 结构。
- ``parse_args``、``main``：解析实验参数并调用通用 workflow。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import nn

from qcomp import (
    CompressionTarget,
    MPOSpec,
    ModelLoadConfig,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
)
from qcomp.evaluation import LMEvalConfig


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
        4096: (16, 16, 16),
        12288: (16, 16, 48),
        151936: (8, 16, 1187),
    }
    if size not in modes:
        raise ValueError(f"no Qwen3 MPO modes configured for feature size {size}")
    return modes[size]


def make_qwen3_mpo_spec(linear: nn.Linear, rank: int) -> MPOSpec:
    """为一个 Qwen3 Linear 创建三核 MPO spec。

    参数：
        linear: 待压缩的 Qwen3 Linear。
        rank: 两条内部 MPO bonds 使用的统一 rank。

    返回：
        与 Linear 输入输出维度匹配的 MPO spec。
    """

    return MPOSpec(
        out_modes=qwen3_modes(linear.out_features),
        in_modes=qwen3_modes(linear.in_features),
        ranks=(1, rank, rank, 1),
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
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--decomposition-provider", default="tensorly")
    parser.add_argument("--execution-provider", default="tensorly")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-layers", type=int)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output")
    parser.add_argument("--log")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    args.task = args.task.strip()
    if not args.task:
        parser.error("--task must not be empty")
    if args.metric is None:
        if args.task != "mmlu":
            parser.error("--metric is required when --task is not mmlu")
        args.metric = ["acc"]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """执行逐层压缩、指定 lm-eval task 评测、恢复和报告输出。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。
    """

    args = parse_args(argv)
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    config = SensitivityExperimentConfig(
        name=f"qwen3-{args.task}-mpo-rank-{args.rank}",
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
            seed=args.seed,
            apply_chat_template=args.apply_chat_template,
        ),
        metrics=tuple(args.metric),
        runtime_config=args.runtime_config,
        decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider,
        start_index=args.start_index,
        max_layers=args.max_layers,
        artifact_root=args.artifact_root,
        output=args.output,
        log=args.log,
    )

    def make_target(path: str, linear: nn.Linear) -> CompressionTarget:
        """将路径 path 与层 linear 转换为本次 rank 对应的 MPO 压缩目标。"""

        return CompressionTarget(
            module_path=path,
            representation="mpo",
            spec=make_qwen3_mpo_spec(linear, args.rank),
        )

    run_sensitivity_experiment(
        config, select_linear=is_target_linear, make_target=make_target
    )


if __name__ == "__main__":
    main()
