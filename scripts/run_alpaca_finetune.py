"""压缩第一个线性层，用 Alpaca 数据集微调张量网络参数，再跑多个 lm-eval task 评测。

流程：
  1. 加载模型 & tokenizer（默认读 runtime.toml）；
  2. 找到模型中第一个无 bias Linear，构造单层 MPO 压缩计划；
  3. compress_model() 将其替换为可训练的 TensorNetworkLinear；
  4. 用 Alpaca 数据集构造训练 DataLoader（tokenize → packing → collate）；
  5. finetune_tensor_network_causal_lm() 只更新张量网络参数；
  6. 用多个 lm-eval task 评测微调后的模型；
  7. 保存训练后 artifact、恢复原始模型、输出总结。

脚本直接调用 qcomp 公共接口，不修改 qcomp 源码，不经过 CLI。

主要内容：
- ``qwen3_modes``、``make_qwen3_mpo_spec``：Qwen3-8B 的 MPO 结构配置。
- ``EVAL_TASKS``：默认评测任务、few-shot 数和关注指标。
- ``parse_args``、``main``：解析参数并执行完整训练-评测流程。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from qcomp import (
    ArtifactPaths,
    CompressionPlan,
    CompressionTarget,
    DataLoaderConfig,
    HuggingFaceDatasetSource,
    MPOSpec,
    ModelLoadConfig,
    TrainingConfig,
    build_causal_lm_dataloader,
    compress_model,
    finetune_tensor_network_causal_lm,
    get_backend,
    list_linears,
    load_causal_lm,
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


# ── 默认评测任务 ─────────────────────────────────────────────────────

EVAL_TASKS: Sequence[tuple[str, int, str]] = (
    ("mmlu",            5,  "acc"),
    ("hellaswag",      10,  "acc_norm"),
    ("winogrande",      5,  "acc"),
    ("arc_challenge",  25,  "acc_norm"),
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
        description="压缩第一个线性层 + Alpaca 微调 + 多数据集评测。"
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
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--decomposition-provider", default="tensorly")
    parser.add_argument("--execution-provider", default="tensorly")
    # 训练数据
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=600)
    # 训练超参
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    # 评测
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-length", type=int, default=4096)
    parser.add_argument(
        "--eval-limit", type=int, default=None,
        help="每个评测 task 的样本上限；省略则全量评测。",
    )
    # 产物
    parser.add_argument("--artifact-root", default="artifacts/alpaca-quick-demo")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)



# ── 主流程 ────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    """执行压缩-微调-评测的完整流程。

    参数：
        argv: 可选命令行参数序列；省略时读取当前进程参数。
    """

    args = parse_args(argv)
    if args.rank <= 0:
        raise ValueError("rank must be positive")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(root=artifact_root)
    paths.create_directories()
    log_path = artifact_root / "experiment.jsonl"

    log_event(
        str(log_path),
        "experiment_started",
        dataset=args.dataset,
        rank=args.rank,
        max_blocks=args.max_blocks,
    )

    # ── 1. 加载模型 ────────────────────────────────────────────────
    print("[1/7] 加载模型 …")
    resources = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=args.model,
            device=args.device,
            dtype=model_dtype,
            trust_remote_code=args.trust_remote_code,
        ),
        runtime_config_path=args.runtime_config,
    )
    model, tokenizer = resources.model, resources.tokenizer
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")


    # ── 2. 找第一个线性层，构造压缩计划 ────────────────────────────
    all_linears = list_linears(model)
    first_path, first_linear = all_linears[0]
    print(
        f"[2/7] 第一个线性层: {first_path}  "
        f"shape=({first_linear.out_features}, {first_linear.in_features})"
    )

    spec = make_qwen3_mpo_spec(first_linear, args.rank)
    target = CompressionTarget(
        module_path=first_path,
        representation="mpo",
        spec=spec,
    )
    plan = CompressionPlan(targets=(target,))

    # ── 3. 压缩（trainable=True 以便后续微调）───────────────────────
    print("[3/7] MPO 分解 & 替换 …")
    decomp_backend = get_backend(args.decomposition_provider, "mpo")
    exec_backend = get_backend(args.execution_provider, "mpo")
    compression_result = compress_model(
        model,
        plan,
        decomposition_backends={"mpo": decomp_backend},
        execution_backends={"mpo": exec_backend},
        trainable=True,
    )

    cm = model_compression_metrics(model, compression_result)
    print(
        f"  压缩率: {cm.compression_ratio:.2f}x  "
        f"({cm.dense_parameters:,} → {cm.compressed_parameters:,} 参数)"
    )
    log_event(
        str(log_path),
        "compressed",
        layer=first_path,
        ratio=cm.compression_ratio,
        dense_params=cm.dense_parameters,
        compressed_params=cm.compressed_parameters,
    )

    # 保存分解 artifact
    for layer_result in compression_result.layer_results:
        artifact_name = f"{first_path.replace('.', '_')}_rank{args.rank}.pt"
        save_artifact(paths.decompositions / artifact_name, layer_result.tn_artifact)


    # ── 4. 构造 Alpaca 训练 DataLoader ─────────────────────────────
    print("[4/7] 构造 Alpaca DataLoader …")
    source = HuggingFaceDatasetSource(dataset=args.dataset, split="train")
    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        max_blocks=args.max_blocks,
        seed=args.seed,
        drop_remainder=True,
        pin_memory=True,
    )
    train_dataloader = build_causal_lm_dataloader(
        source,
        tokenizer,
        dataloader_config,
        text_field=args.text_field,
        runtime_config_path=args.runtime_config,
    )
    print(f"  训练 batch 数: {len(train_dataloader)}")

    # ── 5. 微调（只更新 TensorNetworkLinear 的参数）─────────────────
    print("[5/7] 张量网络参数微调 …")
    training_config = TrainingConfig(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        seed=args.seed,
        device=args.device,
        bf16_autocast=True,
    )
    finetune_result = finetune_tensor_network_causal_lm(
        model,
        train_dataloader,
        training_config,
        output_dir=paths.checkpoints,
    )
    final_loss = finetune_result.training.losses[-1]
    print(
        f"  训练步数: {finetune_result.training.global_step}  "
        f"最终 loss: {final_loss:.4f}  "
        f"可训练参数: {finetune_result.training.trainable_parameters:,}"
    )
    log_event(
        str(log_path),
        "finetune_completed",
        steps=finetune_result.training.global_step,
        final_loss=final_loss,
        trainable_params=finetune_result.training.trainable_parameters,
    )

    # 保存微调后 artifact
    for module_path, artifact in finetune_result.tn_artifacts.items():
        artifact_name = f"finetuned_{module_path.replace('.', '_')}_rank{args.rank}.pt"
        save_artifact(paths.decompositions / artifact_name, artifact)


    # ── 6. 多数据集评测 ────────────────────────────────────────────
    print("[6/7] 评测 …")
    eval_results: dict[str, dict[str, float | int]] = {}
    for task_name, num_fewshot, metric_name in EVAL_TASKS:
        print(f"  → {task_name} ({num_fewshot}-shot, 关注 {metric_name}) …")
        evaluator = LMEvalEvaluator(
            config=LMEvalConfig(
                task=task_name,
                num_fewshot=num_fewshot,
                batch_size=args.eval_batch_size,
                max_length=args.eval_max_length,
                limit=args.eval_limit,
                seed=args.seed,
            ),
            tokenizer=tokenizer,
            runtime_config_path=args.runtime_config,
        )
        result = evaluator(model)
        value = result.metrics.get(metric_name, float("nan"))
        print(f"    {metric_name} = {value:.4f}  (n={result.evaluated_examples})")
        eval_results[task_name] = {
            metric_name: value,
            "examples": result.evaluated_examples,
        }
        log_event(
            str(log_path),
            "eval_completed",
            task=task_name,
            fewshot=num_fewshot,
            **{metric_name: value},
        )

    # ── 7. 恢复原始模型 & 保存总结 ─────────────────────────────────
    print("[7/7] 恢复原始模型 …")
    restore_compressed_model(model, compression_result)
    log_event(str(log_path), "experiment_completed", evaluations=eval_results)

    # 写一份可读的 JSON 总结
    summary = {
        "compressed_layer": first_path,
        "mpo_rank": args.rank,
        "compression_ratio": cm.compression_ratio,
        "dense_parameters": cm.dense_parameters,
        "compressed_parameters": cm.compressed_parameters,
        "train_dataset": args.dataset,
        "max_blocks": args.max_blocks,
        "train_steps": finetune_result.training.global_step,
        "final_loss": final_loss,
        "trainable_parameters": finetune_result.training.trainable_parameters,
        "evaluations": eval_results,
    }
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 终端总结
    print("\n═══ 实验总结 ═══")
    print(f"  压缩层:   {first_path}")
    print(f"  MPO rank: {args.rank}")
    print(f"  压缩率:   {cm.compression_ratio:.2f}x")
    print(f"  训练步数: {finetune_result.training.global_step}")
    print(f"  最终 loss: {final_loss:.4f}")
    for task_name, metrics in eval_results.items():
        print(f"  {task_name}: {metrics}")
    print(f"  产物目录: {artifact_root}")
    print(f"  总结文件: {summary_path}")
    print(f"  实验日志: {log_path}")


if __name__ == "__main__":
    main()
