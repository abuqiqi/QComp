# Evaluation 代码结构

`evaluation` 负责统一测量压缩结果和计算过程，不实现张量分解或模型层收缩。它接收
representations 层定义的通用 artifact、workflows 层产生的模型压缩结果和 backends
层提供的运行能力，输出可以直接记录、比较和汇总的指标。

当前实现包含单个无 bias Linear 的压缩与性能评测、完整 Causal LM 的平均 loss 和
perplexity 评测，以及基于 lm-eval 的 MMLU 总体 accuracy 评测。完整模型性能和 CUDA
显存评测后续接入。

## 目录结构

```text
evaluation/
├── __init__.py       # 导出公开评测接口
├── compression.py    # 参数量、压缩率和重建误差
├── mmlu.py           # 基于 lm-eval 的 MMLU accuracy
├── performance.py    # 分解、推理和训练 step 计时
├── quality.py        # Causal LM 平均 loss 和 perplexity
└── task.py           # 通用评测任务和动态结果
```

主要公共对象：

- `CompressionMetrics`：保存单张量规模、两类压缩率和相对重建误差。
- `compression_metrics`：评测任意已注册张量网络表示的单张量压缩结果。
- `ModelCompressionMetrics`：保存完整模型压缩前后的规模和压缩率。
- `model_compression_metrics`：统计压缩目标和全部未压缩模型参数。
- `EvaluationTask`：用 `requested_metrics` 描述任务需要计算的指标名称。
- `EvaluationResult`：保存任务、动态 metrics、样本数和可选 token 数。
- `evaluate_causal_lm`：按照任务请求的 metrics 评测完整 Causal LM。
- `MMLUEvaluationConfig`：定义 MMLU 执行参数，并通过 `task` 属性提供通用任务描述。
- `evaluate_mmlu`：通过 lm-eval 评测 MMLU 并返回总体 accuracy。
- `TimingResult`：保存多次计时样本，并提供平均值和中位数。
- `time_decomposition`：测量 backend 分解稠密权重的时间。
- `time_inference`：临时构造 Linear 并测量 forward 时间。
- `time_training_step`：临时构造 Linear 并测量完整 SGD step 时间。

## 压缩指标

```python
import torch

from qcomp import MPOSpec, get_backend
from qcomp.evaluation import compression_metrics

weight = torch.randn(4, 4, dtype=torch.float64)
spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
backend = get_backend("native", "mpo")
tn_artifact = backend.decompose(weight, spec)

metrics = compression_metrics(weight, tn_artifact)
print(metrics.compression_ratio)
print(metrics.tensor_size_compression_ratio)
print(metrics.relative_error)
```

参数量直接根据输入计算：

```text
dense_parameters      = weight.numel()
compressed_parameters = tn_artifact 中全部 tensors 的元素数量
compression_ratio     = dense_parameters / compressed_parameters
```

张量字节数会考虑每个张量的数据类型：

```text
dense_tensor_bytes            = weight.numel() × weight.element_size()
compressed_tensor_bytes       = Σ(tensor.numel() × tensor.element_size())
tensor_size_compression_ratio = dense_tensor_bytes / compressed_tensor_bytes
```

参数压缩率只比较标量数量；张量字节压缩率还会反映 FP32、FP16 等数据类型的差异。
这里统计的是张量数据本身，不包含 artifact metadata、磁盘序列化开销、梯度、激活、
optimizer state 或 backend workspace。

相对重建误差定义为：

```text
|| reconstructed - weight ||₂ / || weight ||₂
```

对于矩阵，这也等价于相对 Frobenius 误差。`compression_metrics()` 通过 representations
层的 `reconstruct_tensor()` 根据 `tn_artifact.representation` 自动选择重建函数，不依赖
产生 artifact 的 backend。

```text
tn_artifact.representation
    → representations registry
    → 对应表示的重建函数
    → dense tensor
```

目前 registry 只注册 MPO。新增表示时实现并注册对应的重建函数，evaluation 接口无需
修改。

## 模型级压缩指标

`model_compression_metrics()` 接收仍安装着压缩层的模型和 `compress_model()` 返回的
结果。未压缩部分从模型参数统计，目标层使用原始 Linear 和 canonical artifacts：

```python
from qcomp.evaluation import model_compression_metrics

metrics = model_compression_metrics(model, compression_result)

print(metrics.dense_parameters)
print(metrics.compressed_parameters)
print(metrics.compression_ratio)
print(metrics.tensor_size_compression_ratio)
print(metrics.compressed_layers)
```

这种口径包含 embedding、LayerNorm 和未压缩 Linear 等全部模型参数，同时避免执行
backend 的内部对象影响压缩率。模型恢复后不再使用对应结果计算指标。

## 评测任务与 Causal LM 质量

`EvaluationTask` 决定使用哪个数据集、split、预处理配置和 requested metrics。
具体 evaluator 只计算自己支持的指标，并返回统一的 `EvaluationResult`：

```python
from qcomp.evaluation import EvaluationTask, evaluate_causal_lm

task = EvaluationTask(
    name="wikitext-perplexity",
    dataset="wikitext-2-raw-v1",
    split="test",
    preprocessing="causal-lm-blocks-2048",
    requested_metrics=("loss", "perplexity"),
)
result = evaluate_causal_lm(
    model,
    evaluation_dataloader,
    task,
    device="cuda:0",
)

print(result.task.dataset)
print(result.metrics["loss"])
print(result.metrics["perplexity"])
print(result.evaluated_examples)
print(result.evaluated_tokens)
```

`evaluate_causal_lm()` 接收已经 tokenize 且包含 `labels` 的 DataLoader，按照移位后
不等于 `-100` 的 labels 数量汇总平均 loss；perplexity 等于该 loss 的指数。调用
期间模型处于 inference mode，结束或发生异常后恢复原来的 train/eval 状态。

`EvaluationTask.requested_metrics` 保存准备计算的指标名称；
`EvaluationResult.metrics` 保存实际计算出的指标值。同一个数据集可以建立不同的
`EvaluationTask`，分别请求不同指标；data 层根据任务信息构造 DataLoader。

## MMLU accuracy

`evaluate_mmlu()` 将已经加载的模型和 tokenizer 包装为 lm-eval 的 HFLM。lm-eval
负责 MMLU 数据读取、few-shot prompt、四个选项的概率评分和总体 accuracy 汇总：

```python
from qcomp.evaluation import MMLUEvaluationConfig, evaluate_mmlu

config = MMLUEvaluationConfig(
    num_fewshot=5,
    batch_size=8,
    max_length=4096,
    limit=10,
    seed=42,
)
result = evaluate_mmlu(model, tokenizer, config)

print(result.metrics["accuracy"])
print(result.evaluated_examples)
print(result.task == config.task)
```

当前机器已有的 MMLU 缓存可以直接复用：

```bash
export HF_HOME=/home/xls/workspace/datasets/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

`limit` 接受正整数或 (0, 1) 内的样本比例，并分别作用于每个 MMLU 学科子任务。
敏感性分析可以先设置较小的 limit 验证流程，再使用完整 test split。评测期间模型
切换到 eval mode，结束或异常时恢复原状态。

## 单层性能计时

计时函数由 evaluation 创建临时 Linear、控制 warmup 和重复次数，并在 CUDA 操作前后
同步设备。测量结束后会自动调用 `linear.close()` 释放后端运行资源。

```python
import torch

from qcomp import MPOSpec, get_backend
from qcomp.evaluation import (
    time_decomposition,
    time_inference,
    time_training_step,
)

weight = torch.randn(4, 4)
inputs = torch.randn(3, 4)
targets = torch.randn(3, 4)
spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
backend = get_backend("native", "mpo")
tn_artifact = backend.decompose(weight, spec)

decomposition = time_decomposition(backend, weight, spec)
inference = time_inference(backend, tn_artifact, inputs)
training = time_training_step(backend, tn_artifact, inputs, targets)
```

三个函数的计时边界为：

| 函数 | 计入时间 | 不计入时间 |
|---|---|---|
| `time_decomposition` | 每次 `backend.decompose()` | 调用前的数据准备 |
| `time_inference` | warmup 后的 forward | artifact 移动、Linear 构造、warmup、资源释放 |
| `time_training_step` | 清梯度、forward、MSE、backward、SGD 更新 | artifact 移动、Linear 和 optimizer 构造、warmup、资源释放 |

`TimingResult.samples_seconds` 保存每次正式测量的秒数。不同后端比较时应使用相同权重、
artifact、输入、warmup 和 repeats。

## Backend 能力与环境

评测一个确定可用的后端时可以直接调用计时函数。批量遍历可选后端时，先分别检查其
设计能力和当前环境：

```python
from qcomp import get_backend, list_backends

for provider in list_backends("mpo"):
    backend = get_backend(provider, "mpo")
    if not backend.probe().available:
        continue
    if backend.capabilities.inference:
        result = time_inference(backend, tn_artifact, inputs)
```

`capabilities` 表示后端支持哪些操作，`probe()` 表示依赖和硬件在当前环境中是否可用。
计时函数不会自动切换 backend，也不提供 fallback。

## CUDA 显存评测

CUDA 显存评测尚未实现。后续将沿用计时函数的执行边界，分别测量：

- 分解峰值显存；
- Linear 构造后的驻留显存；
- 首次推理和稳定推理峰值显存；
- 完整训练 step 峰值显存。

PyTorch allocator 指标和进程总 GPU 显存需要分开记录。后者用于覆盖 cuTensorNet 等
计算库自行申请的 workspace，并使用独立测量轮次，避免采样过程影响时间结果。

## 完整模型后续评测

Causal LM 的 loss 和 perplexity 已由 `evaluate_causal_lm()` 提供。后续完整模型
评测继续增加推理吞吐、训练 step、峰值显存、任务 accuracy 和报告汇总；这些功能
仍由 evaluation 管理，不进入具体 backend。
