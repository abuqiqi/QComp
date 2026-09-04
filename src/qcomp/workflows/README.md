# workflows

`workflows` 组合 model、backend、representations 和 training 层，提供面向具体任务的
调用入口。这里负责选择参数和导出任务产物，不重复实现训练循环、张量分解或收缩。

当前包含：

- `compress_linear()`：分解一个无 bias Linear，并安装指定执行 backend 构造的压缩层。
- `compress_model()`：按照 `CompressionPlan` 原子压缩一个或多个模型层。
- `restore_compressed_model()`：关闭计划安装的压缩层并恢复原始 Linear。
- `analyze_sensitivity()`：使用外部 backend 和 evaluator 比较多个临时压缩方案。
- `infer_causal_lm()`：执行正常自回归生成并返回 token、时间、吞吐和显存。
- `finetune_tensor_network_causal_lm()`：选择张量网络参数，调用通用训练层并导出最新 artifacts。

## 模型级压缩

`CompressionTarget` 使用显式模块路径描述一个目标层，并保存该层的 representation 和
spec。`CompressionPlan` 按顺序组合这些目标，因此一个计划既可以全部使用 MPO，也
可以在已有相应实现时混合 MPO、Tucker 或 CP。Provider 和可训练状态不写入计划。

```python
from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    compress_model,
    get_backend,
    restore_compressed_model,
)

plan = CompressionPlan(
    targets=(
        CompressionTarget(
            "model.layers.0.self_attn.q_proj",
            "mpo",
            MPOSpec(out_modes=(32, 128), in_modes=(32, 128), ranks=(1, 8, 1)),
        ),
        CompressionTarget(
            "model.layers.0.self_attn.k_proj",
            "mpo",
            MPOSpec(out_modes=(8, 128), in_modes=(32, 128), ranks=(1, 8, 1)),
        ),
    ),
)
backend = get_backend("native", "mpo")
result = compress_model(
    model,
    plan,
    decomposition_backends={"mpo": backend},
    execution_backends={"mpo": backend},
    trainable=False,
)

restore_compressed_model(model, result)
```

`result.layer_results` 与目标顺序一致，每项包含该层的 artifact 和替换记录。执行前会
确认全部目标路径和 representation 对应的 backend；任一层失败时恢复本次已经替换的
所有层。同一种 representation 在一次调用中共用一对外部 backend。

## 压缩敏感性分析

`SensitivityCase` 用名称和一个 `CompressionPlan` 描述一次实验。计划可以只包含一个
Linear，也可以包含多个层。`analyze_sensitivity()` 先评测未压缩 baseline，再逐 case
临时压缩、统计逐层及整模压缩指标、执行 evaluator，并始终恢复原始 Linear。

```python
from functools import partial

from qcomp import SensitivityCase, analyze_sensitivity, get_backend
from qcomp.evaluation import MMLUEvaluationConfig, evaluate_mmlu

tensorly_mpo_backend = get_backend("tensorly", "mpo")
mmlu_evaluator = partial(
    evaluate_mmlu,
    tokenizer=tokenizer,
    config=MMLUEvaluationConfig(limit=1),
)
result = analyze_sensitivity(
    model,
    cases=(SensitivityCase("candidate-mpo", plan),),
    decomposition_backends={"mpo": tensorly_mpo_backend},
    execution_backends={"mpo": tensorly_mpo_backend},
    evaluator=mmlu_evaluator,
    metric_directions={"accuracy": "higher"},
)
```

Evaluator 是接收当前模型并返回 `EvaluationResult` 的函数；tokenizer、DataLoader 和
数据集配置由该函数在外部固定。原始指标保存在每个 `evaluation.metrics` 中，
`metric_degradations` 保存指定指标相对 baseline 的退化量。`higher` 指标使用
`baseline - compressed`，`lower` 指标使用 `compressed - baseline`，因此正值统一
表示性能下降。

## 正常 Causal LM 推理

`infer_causal_lm()` 接收完整模型和已经 tokenize 的 DataLoader，逐 batch 调用模型
自身的 `generate()`。模型加载与设备迁移不计时；`total_seconds` 包含 DataLoader
迭代、batch 搬运、生成和输出回传 CPU，`generation_seconds` 只包含同步后的
`generate()` 调用。

```python
from qcomp import InferenceConfig, infer_causal_lm

result = infer_causal_lm(
    model,
    inference_dataloader,
    InferenceConfig(
        device="cuda:0",
        max_new_tokens=64,
        do_sample=False,
    ),
)

generated_batches = result.generated_token_ids
performance = result.performance
print(performance.total_seconds)
print(performance.generation_seconds)
print(performance.generated_tokens_per_second)
print(performance.peak_allocated_bytes)
```

`generated_token_ids` 只保存相对于输入新增的 token，并在返回前移动到 CPU。输入
`labels` 不传给 `generate()`。CUDA 显存字段来自 PyTorch allocator；CPU 推理时
为 `None`，且不包含 cuTensorNet 等库绕过 PyTorch allocator 申请的显存。

这些 token 可以在后续使用 tokenizer 解码并计算生成任务 metrics；PPL 和 loss 继续
使用 `evaluation.evaluate_causal_lm()`，模型压缩率继续由 compression 评测负责。

## 张量网络微调

该 workflow 接收已经压缩的模型和产生 tokenized batch 的 `DataLoader`。它查找构造
时设置为 `trainable=True` 的 `TensorNetworkLinear` 参数，然后使用
`CausalLMObjective` 调用 `training.train_causal_lm()`。

```python
from qcomp import TrainingConfig, finetune_tensor_network_causal_lm

result = finetune_tensor_network_causal_lm(
    model,
    train_dataloader,
    TrainingConfig(
        max_steps=100,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        save_steps=25,
        device="cuda:0",
    ),
    "artifacts/checkpoints/run-1",
)
```

返回值将通用训练结果和张量网络特有产物分开保存：

```python
result.training.global_step
result.training.checkpoint_path
result.module_paths
result.tn_artifacts
```

继续训练时，需要先构造相同的压缩模型结构和执行 backend，再传入明确的 checkpoint：

```python
result = finetune_tensor_network_causal_lm(
    model,
    train_dataloader,
    config,
    "artifacts/checkpoints/run-1-resumed",
    resume_from="artifacts/checkpoints/run-1/checkpoint-25.pt",
)
```

checkpoint 只保存本次选择的参数、优化器、调度器、训练位置和随机数状态，不重复保存
冻结参数。DataLoader 需要可重复迭代并保持确定的数据顺序。

## 增加其他微调方法

新的微调 workflow 负责选择自己的参数和 objective，然后调用同一个
`train_causal_lm()`。例如 Cayley workflow 将选择 Cayley Adapter 参数并使用包含
teacher 的知识蒸馏 objective；训练循环和 checkpoint 仍由 `training` 层提供。
