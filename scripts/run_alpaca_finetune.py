"""联合压缩 Qwen3 指定 block 范围内的全部投影层，并用 Alpaca 微调 MPO 参数。

本脚本以零起始、包含起点的 block 编号筛选 attention 与 MLP 投影层，调用公共
压缩、数据、微调和评测接口完成原模型、联合压缩后、微调后三阶段对比。默认从
``model.layers.25`` 开始，rank 为 96；训练冻结其余参数，支持同一训练计划的断点续训。
产物包含逐层 artifact、训练 checkpoint、实验配置和评测总结。

主要内容：
- ``qwen3_modes``、``make_qwen3_mpo_spec``：定义 Qwen3-8B 的 MPO 结构。
- ``make_compression_plan``：按 block 范围选择全部 attention 与 MLP 投影层。
- ``evaluate_stage``：复用相同评测配置比较各阶段模型。
- ``save_layer_artifacts``：按模块路径分别保存压缩产物。
- ``parse_args``、``main``：解析配置并编排联合压缩、微调和评测。
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .qwen3_mpo_config import qwen3_modes, qwen3_mpo_spec_dict
else:
    from qwen3_mpo_config import qwen3_modes, qwen3_mpo_spec_dict

from qcomp import (
    ArtifactPaths,
    CompressionPlan,
    CompressionTarget,
    DataLoaderConfig,
    HuggingFaceDatasetSource,
    MPOSpec,
    ModelLoadConfig,
    TensorNetworkArtifact,
    TrainingConfig,
    build_causal_lm_dataloader,
    compress_model,
    finetune_tensor_network_causal_lm,
    get_backend,
    list_linears,
    load_causal_lm,
    load_runtime_config,
    log_event,
    restore_compressed_model,
    save_artifact,
)
from qcomp.evaluation import (
    LMEvalConfig,
    LMEvalEvaluator,
    model_compression_metrics,
)


# ── Qwen3-8B MPO 结构 ────────────────────────────────────────────────


def make_qwen3_mpo_spec(linear: nn.Linear, rank: int) -> MPOSpec:
    """为一个 Qwen3 Linear 创建三核 MPO spec。

    参数：
        linear: 待压缩的 Qwen3 Linear。
        rank: 两条内部 MPO bonds 使用的统一 rank。

    返回：
        与 Linear 输入输出维度匹配的 MPO spec。
    """

    return MPOSpec(**qwen3_mpo_spec_dict(linear.out_features, linear.in_features, rank))


# ── 默认评测任务 ─────────────────────────────────────────────────────

EVAL_TASKS: Sequence[tuple[str, int, str]] = (
    ("mmlu", 5, "acc"),
    ("hellaswag", 10, "acc_norm"),
)

# ── 命令行参数 ────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析模型、压缩、训练和评测参数。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。

    返回：
        已补全默认值的参数命名空间。
    """

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="联合压缩 Qwen3 指定 block 的全部 proj，再用 Alpaca 微调 MPO 参数并评测。",
    )
    # 模型
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
    # 压缩
    parser.add_argument(
        "--start-block",
        type=int,
        default=25,
        help="起始 Transformer block 编号（从 0 计数，包含；默认 25）。",
    )
    parser.add_argument(
        "--end-block",
        type=int,
        help="结束 Transformer block 编号（不包含）；省略则到最后一个 block。",
    )
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--decomposition-provider", default="tensorly")
    parser.add_argument("--execution-provider", default="tensorly")
    # 训练数据
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-token-blocks",
        type=int,
        help="训练 token block 数量上限；省略则使用全部训练文本。",
    )
    # 训练超参
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        help="同一模型、数据、压缩结构及训练配置的 checkpoint 路径。",
    )
    # 评测
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-length", type=int, default=4096)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="每个评测 task 的样本上限；省略则全量评测。",
    )
    # 产物
    parser.add_argument(
        "--artifact-root",
        help="直接指定产物目录；默认按模型、block 范围、rank 和时间戳自动生成。",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if args.rank <= 0:
        parser.error("--rank must be positive")
    if args.start_block < 0:
        parser.error("--start-block must be non-negative")
    if args.end_block is not None and args.end_block <= args.start_block:
        parser.error("--end-block must be greater than --start-block")
    if args.resume_from is not None and not Path(args.resume_from).is_file():
        parser.error("--resume-from must point to an existing checkpoint")
    return args


def make_compression_plan(
    model: nn.Module,
    start_block: int,
    end_block: int | None,
    rank: int,
) -> CompressionPlan:
    """选择指定 Qwen3 block 范围内的全部无 bias 投影层。

    参数：
        model: 具有 ``model.layers`` 的 Qwen3 Causal LM。
        start_block: 从零开始、包含的起始 block 编号。
        end_block: 不包含的结束编号；None 表示模型末尾。
        rank: 所有目标共享的 MPO 内部 bond rank。

    返回：
        包含 attention 的 q/k/v/o_proj 和 MLP 的 gate/up/down_proj 的联合压缩计划。

    异常：
        ValueError: 范围越界、rank 非正数或没有匹配投影层。
    """

    block_count = len(model.get_submodule("model.layers"))
    stop = block_count if end_block is None else end_block
    if not 0 <= start_block < stop <= block_count:
        raise ValueError(f"block range must satisfy 0 <= start < end <= {block_count}")
    if rank <= 0:
        raise ValueError("rank must be positive")
    pattern = re.compile(
        r"model\.layers\.(\d+)\."
        r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
    )
    targets = []
    for module_path, linear in list_linears(model):
        match = pattern.fullmatch(module_path)
        if match is not None and start_block <= int(match.group(1)) < stop:
            targets.append(
                CompressionTarget(
                    module_path=module_path,
                    representation="mpo",
                    spec=make_qwen3_mpo_spec(linear, rank),
                )
            )
    if not targets:
        raise ValueError(
            "the selected block range contains no supported projection layers"
        )
    return CompressionPlan(targets=tuple(targets))


def evaluate_stage(
    model: nn.Module,
    evaluators: Mapping[str, LMEvalEvaluator],
    stage: str,
    log_path: Path,
) -> dict[str, dict[str, float | int]]:
    """用相同 evaluator 评测一个阶段，并逐 task 保存日志。

    参数：
        model: 当前阶段的完整语言模型。
        evaluators: 按 task 名称保存、跨阶段复用的 evaluator。
        stage: baseline、compressed 或 finetuned。
        log_path: JSONL 实验日志路径。

    返回：
        各 task 的关注指标与实际评测样本数。
    """

    results = {}
    for task_name, num_fewshot, metric_name in EVAL_TASKS:
        print(f"  [{stage}] {task_name} ({num_fewshot}-shot) ...", flush=True)
        result = evaluators[task_name](model)
        value = result.metrics[metric_name]
        results[task_name] = {
            metric_name: value,
            "examples": result.evaluated_examples,
        }
        print(
            f"    {metric_name} = {value:.4f} (n={result.evaluated_examples})",
            flush=True,
        )
        log_event(
            log_path,
            "eval_completed",
            stage=stage,
            task=task_name,
            fewshot=num_fewshot,
            **results[task_name],
        )
    return results


def save_layer_artifacts(
    artifacts: Mapping[str, TensorNetworkArtifact],
    directory: Path,
    rank: int,
) -> dict[str, str]:
    """为每个模块保存独立 artifact，并返回模块到文件的映射。

    参数：
        artifacts: 以完整模块路径为键的 artifact 集合。
        directory: 当前阶段的产物目录。
        rank: 文件名中记录的 MPO rank。

    返回：
        完整模块路径到 artifact 文件路径的映射。
    """

    paths = {}
    for module_path, artifact in artifacts.items():
        path = directory / f"{module_path.replace('.', '_')}_rank{rank}.pt"
        save_artifact(path, artifact)
        paths[module_path] = str(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """执行原模型评测、联合压缩、压缩后评测、MPO 微调和最终评测。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。
    """

    args = parse_args(argv)
    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        max_blocks=args.max_token_blocks,
        seed=args.seed,
        drop_remainder=True,
        pin_memory=args.device.startswith("cuda"),
    )
    training_config = TrainingConfig(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        seed=args.seed,
        device=args.device,
        bf16_autocast=args.model_dtype == "bfloat16",
    )
    evaluation_configs = {
        task: LMEvalConfig(
            task=task,
            num_fewshot=fewshot,
            batch_size=args.eval_batch_size,
            max_length=args.eval_max_length,
            limit=args.eval_limit,
            evaluation_seed=args.seed,
        )
        for task, fewshot, _ in EVAL_TASKS
    }
    runtime = load_runtime_config(args.runtime_config)
    model_config = ModelLoadConfig(
        model_name_or_path=args.model or runtime.model_name_or_path,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[args.model_dtype],
        trust_remote_code=args.trust_remote_code,
    )
    print("[1/7] 加载模型并选择投影层 ...", flush=True)
    resources = load_causal_lm(model_config, runtime_config_path=args.runtime_config)
    model, tokenizer = resources.model, resources.tokenizer
    plan = make_compression_plan(model, args.start_block, args.end_block, args.rank)
    module_paths = tuple(target.module_path for target in plan.targets)
    end_block = (
        args.end_block
        if args.end_block is not None
        else len(model.get_submodule("model.layers"))
    )
    print(
        f"  block 编号 [{args.start_block}, {end_block})，"
        f"共 {len(module_paths)} 个 proj，MPO rank {args.rank}",
        flush=True,
    )
    for module_path in module_paths:
        print(f"    {module_path}")

    if args.artifact_root is None:
        model_name = Path(model_config.model_name_or_path).name
        run_name = (
            f"{model_name}_blocks-{args.start_block}-{end_block}_rank-{args.rank}"
        )
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        args.artifact_root = str(
            Path("artifacts/alpaca-finetune") / run_name / timestamp
        )
    paths = ArtifactPaths(root=Path(args.artifact_root))
    print(f"  产物目录: {paths.root}", flush=True)
    paths.create_directories()
    log_path = paths.root / "experiment.jsonl"
    experiment_config = {
        **vars(args),
        "model": model_config.model_name_or_path,
        "end_block": end_block,
        "compressed_layers": module_paths,
        "trainable_scope": "mpo",
        "decomposition_dtype": "float32",
    }
    (paths.root / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_event(log_path, "experiment_started", **experiment_config)
    evaluators = {
        task: LMEvalEvaluator(
            tokenizer,
            config,
            runtime_config_path=args.runtime_config,
        )
        for task, config in evaluation_configs.items()
    }
    print("[2/7] 原模型评测 ...", flush=True)
    evaluations = {"baseline": evaluate_stage(model, evaluators, "baseline", log_path)}

    print("[3/7] 联合分解与替换全部目标 proj ...", flush=True)
    compression_result = compress_model(
        model,
        plan,
        decomposition_backends={"mpo": get_backend(args.decomposition_provider, "mpo")},
        execution_backends={"mpo": get_backend(args.execution_provider, "mpo")},
        trainable=True,
        decomposition_dtype=torch.float32,
    )
    try:
        cm = model_compression_metrics(model, compression_result)
        print(
            f"  模型参数压缩率 {cm.compression_ratio:.2f}x "
            f"({cm.dense_parameters:,} → {cm.compressed_parameters:,})",
            flush=True,
        )
        log_event(
            log_path,
            "compressed",
            layers=module_paths,
            ratio=cm.compression_ratio,
            dense_params=cm.dense_parameters,
            compressed_params=cm.compressed_parameters,
        )
        initial_artifacts = save_layer_artifacts(
            {
                target.module_path: result.tn_artifact
                for target, result in zip(
                    plan.targets, compression_result.layer_results, strict=True
                )
            },
            paths.decompositions / "initial",
            args.rank,
        )
        print("[4/7] 联合压缩后、微调前评测 ...", flush=True)
        evaluations["compressed"] = evaluate_stage(
            model, evaluators, "compressed", log_path
        )

        print("[5/7] 构造 Alpaca DataLoader ...", flush=True)
        train_dataloader = build_causal_lm_dataloader(
            HuggingFaceDatasetSource(dataset=args.dataset, split="train"),
            tokenizer,
            dataloader_config,
            text_field=args.text_field,
            runtime_config_path=args.runtime_config,
        )
        print(f"  训练 batch 数: {len(train_dataloader)}", flush=True)
        print("[6/7] 微调全部压缩层的 MPO 参数，其余参数冻结 ...", flush=True)
        finetune_result = finetune_tensor_network_causal_lm(
            model,
            train_dataloader,
            training_config,
            output_dir=paths.checkpoints,
            resume_from=args.resume_from,
        )
        training = finetune_result.training
        # 已完成的 checkpoint 可以直接恢复并导出，此时本次没有新增 loss。
        final_loss = training.losses[-1] if training.losses else None
        final_artifacts = save_layer_artifacts(
            finetune_result.tn_artifacts,
            paths.decompositions / "finetuned",
            args.rank,
        )
        log_event(
            log_path,
            "finetune_completed",
            steps=training.global_step,
            final_loss=final_loss,
            trainable_params=training.trainable_parameters,
            checkpoint=str(training.checkpoint_path),
        )
        print("[7/7] 微调后评测 ...", flush=True)
        evaluations["finetuned"] = evaluate_stage(
            model, evaluators, "finetuned", log_path
        )
        summary = {
            "model": model_config.model_name_or_path,
            "start_block": args.start_block,
            "end_block": end_block,
            "compressed_layers": module_paths,
            "mpo_rank": args.rank,
            "trainable_scope": "mpo",
            "compression_ratio": cm.compression_ratio,
            "dense_parameters": cm.dense_parameters,
            "compressed_parameters": cm.compressed_parameters,
            "train_dataset": args.dataset,
            "max_token_blocks": args.max_token_blocks,
            "train_steps": training.global_step,
            "final_loss": final_loss,
            "trainable_parameters": training.trainable_parameters,
            "checkpoint": str(training.checkpoint_path),
            "initial_artifacts": initial_artifacts,
            "finetuned_artifacts": final_artifacts,
            "evaluations": evaluations,
        }
        summary_path = paths.root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log_event(log_path, "experiment_completed", summary=str(summary_path))
        print(
            f"  训练步数: {training.global_step}，本次最终 loss: {final_loss}",
            flush=True,
        )
        print(f"  三阶段评测总结: {summary_path}", flush=True)
        print(f"  训练 checkpoint: {training.checkpoint_path}", flush=True)
    finally:
        restore_compressed_model(model, compression_result)


if __name__ == "__main__":
    main()
