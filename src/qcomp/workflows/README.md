# workflows

`workflows` 组合 model、backend、representations 和 training 层，提供面向具体任务的调用入口。这里负责选择参数和导出任务产物。

当前包含：

- `compress_linear()`：分解一个无 bias Linear，并安装指定执行 backend 构造的压缩层。
- `compress_model()`：按照 `CompressionPlan` 原子压缩一个或多个模型层。
- `restore_compressed_model()`：关闭计划安装的压缩层并恢复原始 Linear。
- `SensitivityExperimentConfig`、`run_sensitivity_experiment()`：配置资源并执行逐层 lm-eval 实验，统一输出日志和报告。
- `analyze_sensitivity()`：使用外部 backend 和 evaluator 比较多个临时压缩方案。
- `format_sensitivity_report()`：将敏感性分析结果转换为 Markdown，并按需写入文件。
- `infer_causal_lm()`：执行正常自回归生成并返回 token、时间、吞吐和显存。
- `finetune_tensor_network_causal_lm()`：选择张量网络参数，调用通用训练层并导出最新 artifacts。

## 模型级压缩

`CompressionTarget` 使用显式模块路径描述一个目标层，并保存该层的 representation 和 spec。`CompressionPlan` 按顺序组合这些目标，因此一个计划既可以全部使用 MPO，也可以在已有相应实现时混合 MPO、Tucker 或 CP。Provider 和可训练状态不写入计划。

下例先加载默认模型，其目标路径和维度采用 Qwen3-8B 配置：

```python
import torch
from qcomp import (
    load_causal_lm,
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    compress_model,
    get_backend,
    restore_compressed_model,
)

resources = load_causal_lm()
model, tokenizer = resources.model, resources.tokenizer
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
    decomposition_dtype=torch.float32,
)

restore_compressed_model(model, result)
```

`result.layer_results` 与目标顺序一致，每项包含该层的 artifact 和替换记录。执行前会确认全部目标路径和 representation 对应的 backend；任一层失败时恢复本次已经替换的所有层。同一种 representation 在一次调用中共用一对外部 backend。

三个公共入口 `compress_linear()`、`compress_model()` 和 `analyze_sensitivity()` 均接受 `decomposition_dtype`。例如设置 `decomposition_dtype=torch.float32`，会以 FP32 分解权重，并在构造压缩层前将 artifact 转回原层的设备和 dtype。这些底层入口默认 `None` 使用原权重类型；上层 `SensitivityExperimentConfig.decomposition_dtype` 默认 FP32。backend 直接使用 `get_backend()` 返回的对象。

## 压缩敏感性分析

`SensitivityCase` 用名称和一个 `CompressionPlan` 描述一次实验。计划可以只包含一个 Linear，也可以包含多个层。`analyze_sensitivity()` 先评测未压缩 baseline，再逐 case 临时压缩、统计逐层及整模压缩指标、执行 evaluator，并始终恢复原始 Linear。

接续上例已经恢复的 `model`、`tokenizer` 和 `plan`：

```python
from qcomp import (
    ArtifactPaths,
    SensitivityCase,
    analyze_sensitivity,
    format_sensitivity_report,
    get_backend,
    resolve_metric_directions,
)
from qcomp.evaluation import LMEvalConfig, LMEvalEvaluator

tensorly_mpo_backend = get_backend("tensorly", "mpo")
benchmark_evaluator = LMEvalEvaluator(
    tokenizer,
    LMEvalConfig(task="mmlu", limit=1),
)
paths = ArtifactPaths()
result = analyze_sensitivity(
    model,
    cases=(SensitivityCase("candidate-mpo", plan),),
    decomposition_backends={"mpo": tensorly_mpo_backend},
    execution_backends={"mpo": tensorly_mpo_backend},
    evaluator=benchmark_evaluator,
    decomposition_dtype=torch.float32,
    metric_directions=resolve_metric_directions(("acc",)),
)
report = format_sensitivity_report(
    result,
    paths.root / "sensitivity" / "qwen3-mmlu-sensitivity.md",
)
```

Evaluator 是接收当前模型并返回 `EvaluationResult` 的可调用对象。`LMEvalEvaluator` 在第一次调用时加载配置指定的 task 或 group，后续 case 复用同一任务对象，只重新执行当前模型；更换标准 benchmark 时修改 `LMEvalConfig.task`，并同步选择该任务返回的指标、更新 `metric_directions`。例如 GSM8K 使用 `resolve_metric_directions(("exact_match_strict_match",))`；上层实验入口对应设置 `SensitivityExperimentConfig.metrics=("exact_match_strict_match",)`，脚本对应传入 `--metric exact_match_strict_match`。原始指标保存在每个 `evaluation.metrics` 中，`metric_degradations` 保存指定指标相对 baseline 的退化量。`higher` 指标使用 `baseline - compressed`，`lower` 指标使用 `compressed - baseline`，因此正值统一表示性能下降。`format_sensitivity_report()` 始终返回 Markdown；指定输出路径时会创建父目录并写入相同内容。需要实时处理逐个 case 的结果时，可通过 `on_case_result` 传入回调；回调在该 case 恢复原模型后执行。`sensitivity_case_record(result)` 将单次结果整理为包含指标和耗时的独立字典，由调用方交给 `log_event(path, "case_completed", **record)` 或其他输出接口。

### 逐层实验入口

`SensitivityExperimentConfig` 复用 `ModelLoadConfig` 和 `LMEvalConfig`，配置实验名称、指标、backend、分段和输出。`run_sensitivity_experiment()` 加载模型，先用 `select_linear(path, linear)` 筛选，再应用 `start_layer_index` 和 `max_layers`，最后用 `make_target(path, linear)` 为每层生成一个 `CompressionTarget`，包装成独立 case。回调必须保留选中层的模块路径。backend 根据实际 target 的 representation 创建。

```python
from qcomp import (
    CompressionTarget, MPOSpec, ModelLoadConfig,
    SensitivityExperimentConfig, run_sensitivity_experiment,
)
from qcomp.evaluation import LMEvalConfig

config = SensitivityExperimentConfig(
    name="linear-full-rank", model=ModelLoadConfig(device="cpu"),
    evaluation=LMEvalConfig(task="mmlu", limit=1), metrics=("acc",),
    decomposition_provider="native", execution_provider="native", max_layers=1,
)

def make_target(path, linear):
    """将路径 path 与层 linear 构造为单核满秩目标。"""
    return CompressionTarget(
        path, "mpo", MPOSpec.full_rank((linear.out_features,), (linear.in_features,)),
    )

result = run_sensitivity_experiment(
    config, select_linear=lambda path, linear: path != "lm_head", make_target=make_target,
)
```

输出默认集中在 `artifacts/sensitivity/<name>/<timestamp>/`，报告为 `layers-<start>-<last>.md`，日志为同名 `.jsonl`。实验名称自动处理为安全的目录名，时间戳采用运行开始时的 UTC+8 时间，格式为 `YYYYMMDDTHHMMSS`（精确到秒）；`output` 和 `log` 可以覆盖路径。目录按需创建，日志追加写入，每次运行的开始、case 与完成记录共享 `run_id`。开始记录包含模型来源、backend、dtype 和评测配置。入口返回 `SensitivityResult`；case 使用模块路径命名，实验名称可包含 rank。

Qwen3 脚本保留命令行参数、层选择规则和 MPO spec 构造，调用上层入口完成执行与输出。多层联合方案或自定义 evaluator 继续使用底层 `analyze_sensitivity()`。

## 正常 Causal LM 推理

`infer_causal_lm()` 接收完整模型和已经 tokenize 的 DataLoader，逐 batch 调用模型自身的 `generate()`。模型加载与设备迁移不计时；`total_seconds` 包含 DataLoader 迭代、batch 搬运、生成和输出回传 CPU，`generation_seconds` 只包含同步后的 `generate()` 调用。

以下假设 `model` 已加载，`inference_dataloader` 产生二维 `input_ids` 和可选 `attention_mask`；生成输入应按 tokenizer 的生成约定准备。

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

`generated_token_ids` 只保存相对于输入新增的 token，并在返回前移动到 CPU。输入 `labels` 不传给 `generate()`。CUDA 显存字段来自 PyTorch allocator；CPU 推理时为 `None`，且不包含 cuTensorNet 等库绕过 PyTorch allocator 申请的显存。

`generated_tokens` 按新增输出张量的元素数量统计，包含其中的 EOS 或 padding，不单独过滤提前结束序列的填充 token。

这些 token 可以在后续使用 tokenizer 解码；标准模型质量使用 `LMEvalEvaluator`，模型压缩率继续由 compression 评测负责。

## 张量网络微调

该 workflow 接收已经压缩的模型和产生 tokenized batch 的 `DataLoader`。它查找构造时或后续设置为 `requires_grad=True` 的 `TensorNetworkLinear` 参数，然后使用 `CausalLMObjective` 调用 `training.train_causal_lm()`。

以下假设 `model` 已通过 `trainable=True` 安装压缩层，`train_dataloader` 已准备好：

```python
from qcomp import TrainingConfig, finetune_tensor_network_causal_lm

config = TrainingConfig(
    max_steps=100, gradient_accumulation_steps=8, learning_rate=1e-5,
    save_steps=25, device="cuda:0",
)
result = finetune_tensor_network_causal_lm(
    model,
    train_dataloader,
    config,
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

`tn_artifacts` 是返回的内存对象；需要保存时由调用方使用 `save_artifact()` 写入文件。

checkpoint 只保存本次选择的参数、优化器、调度器、训练位置和随机数状态，不重复保存冻结参数。DataLoader 需要可重复迭代并保持确定的数据顺序。

## 增加其他微调方法

参数选择、objective、workflow 和 checkpoint 的扩展步骤见[扩展指南](../../../docs/extending.md#新增训练方法)。

## 选层 JSON 计划

`load_compression_plan(path, model=model)` 读取统一选层 JSON 的执行部分，并自动检查目标无 bias Linear 和矩阵维度，成功后返回 `CompressionPlan`。它不读取敏感度来源、不修改模型。`compression_plan_to_dict()` 和 `compression_plan_from_dict()` 读写 `compression_plan` 对象，支持逐矩阵不同 modes 与 ranks 的 MPO spec。文件结构和使用示例见[实验文档](../../../docs/experiments.md#统一-json-与计划加载)。
