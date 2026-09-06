# evaluation

`evaluation` 负责统一测量压缩结果和计算过程，不实现张量分解或模型层收缩。它接收
representations 层定义的通用 artifact、workflows 层产生的模型压缩结果和 backends
层提供的运行能力，输出可以直接记录、比较和汇总的指标。

当前实现包含单张量与完整模型的压缩指标、单个无 bias Linear 的独立性能 benchmark，以及基于 lm-eval 的
通用标准 benchmark 评测。完整模型正常
生成的时间、吞吐和 PyTorch CUDA 峰值显存由 `workflows/inference.py` 在真实推理过程
中记录。

## 目录结构

```text
evaluation/
├── __init__.py       # 导出公开评测接口
├── compression.py    # 参数量、压缩率和重建误差
├── lm_eval.py        # 通用 lm-eval task/group evaluator
├── metrics.py        # 常用评测指标的优化方向
├── performance.py    # 分解、推理和训练 step 计时
└── task.py           # 通用评测任务和动态结果
```

主要公共对象：

- `CompressionMetrics`：保存单张量规模、两类压缩率和相对重建误差。
- `compression_metrics`：评测任意已注册张量网络表示的单张量压缩结果。
- `ModelCompressionMetrics`：保存完整模型压缩前后的规模和压缩率。
- `model_compression_metrics`：统计压缩目标和全部未压缩模型参数。
- `EvaluationTask`：用 `requested_metrics` 描述任务需要计算的指标名称。
- `EvaluationResult`：保存任务、动态 metrics、样本数和可选 token 数。
- `LMEvalConfig`：定义任一 lm-eval task 或 group 的执行参数。
- `LMEvalEvaluator`：加载一次 task，并重复评测不同模型状态，动态返回 task metrics。
- `MetricDirection`：限定指标是数值越高还是越低越好。
- `metric_direction`、`resolve_metric_directions`：查询常用指标方向。
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
结果（下例的 `model` 和 `compression_result` 由该调用提供）。未压缩部分从模型参数统计，目标层使用原始 Linear 和 canonical artifacts：

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

## 指标方向

`metrics.py` 统一保存常用指标的优化方向，不维护 task 到主指标的映射。调用方选择
需要比较的指标，再把解析结果传给 sensitivity workflow：

```python
from qcomp.evaluation import metric_direction, resolve_metric_directions

assert metric_direction("acc") == "higher"
directions = resolve_metric_directions(("acc", "word_perplexity"))
```

当前注册 accuracy、exact match、F1、loss、perplexity 和 bits-per-byte 等常用名称。
未知指标会明确报错，避免静默采用错误方向。

## lm-eval 标准 benchmark

`LMEvalEvaluator` 将模型和 tokenizer 包装为 lm-eval 的 HFLM。task 名称不绑定到
qcomp 类：MMLU、HellaSwag、BoolQ、GSM8K 等 lm-eval 已注册 task 或 group 都复用同一个
evaluator，只由 task 定义 prompt、request、filter 和 metrics。

先使用 `load_causal_lm()` 取得 `resources`，再构造 evaluator：

```python
from qcomp import load_causal_lm
from qcomp.evaluation import LMEvalConfig, LMEvalEvaluator

resources = load_causal_lm()
model, tokenizer = resources.model, resources.tokenizer

evaluator = LMEvalEvaluator(
    tokenizer,
    LMEvalConfig(
        task="mmlu",
        num_fewshot=5,
        batch_size=8,
        max_length=4096,
        limit=10,
        seed=42,
    ),
)
result = evaluator(model)

print(result.metrics)
print(result.evaluated_examples)
```

第一次调用会加载 lm-eval 的 task/group 定义，数据读取使用 `config/runtime.toml` 配置的 Hugging Face cache，
并启用 lm-eval request cache；后续调用复用相同 task 和已缓存 request，只重新评测当前
模型状态。默认 TOML 设置
`offline = true`，因此不会检查 Hub；若确实要允许联网，使用另一份明确设置
`offline = false` 的 runtime TOML。

这里不传入 `qcomp.data` 构造的训练 DataLoader。lm-eval 自己负责标准 benchmark 的
数据、prompt、few-shot、request batching、答案 filter 和 metric 聚合。评测期间模型
切换到 eval mode，完成或异常时恢复原状态。

`limit` 接受正整数、(0, 1) 内的样本比例或 `None`（不限制）；`num_fewshot=None` 使用 task 默认值。结果指标名由 lm-eval 动态转换，例如
`acc,none` 变为 `acc`，无需为每个数据集新增 evaluator 文件。

## 单层性能计时

计时函数由 evaluation 控制 warmup 和重复次数，并在 CUDA 操作前后同步设备。
推理和训练 step 测量结束后会自动调用 `linear.close()` 释放临时 Linear 的后端资源。

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

`time_decomposition()` 不构造 Linear；只有推理和训练 step 计时会创建并关闭临时 Linear。

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
from qcomp.evaluation import time_inference

for provider in list_backends("mpo"):
    backend = get_backend(provider, "mpo")
    if not backend.probe().available:
        continue
    if backend.capabilities.inference:
        runtime_inputs = inputs.to("cuda" if provider == "cutensornet" else inputs.device)
        result = time_inference(backend, tn_artifact, runtime_inputs)
```

`capabilities` 表示后端支持哪些操作，`probe()` 表示依赖和硬件在当前环境中是否可用。
计时函数不会自动切换 backend，也不提供 fallback。

## CUDA 显存边界

`infer_causal_lm()` 已记录一次真实完整模型生成的 PyTorch peak allocated/reserved
显存。后续独立 backend benchmark 将沿用计时函数的执行边界，分别测量：

- 分解峰值显存；
- Linear 构造后的驻留显存；
- 首次推理和稳定推理峰值显存；
- 完整训练 step 峰值显存。

PyTorch allocator 指标和进程总 GPU 显存需要分开记录。后者用于覆盖 cuTensorNet 等
计算库自行申请的 workspace，并使用独立测量轮次，避免采样过程影响时间结果。

## 完整模型后续评测

标准模型质量由 `LMEvalEvaluator` 评测，正常 inference workflow 返回真实生成吞吐和
PyTorch 峰值显存。后续完整模型 benchmark 继续增加重复生成、训练 step、进程级峰值
显存和跨阶段报告汇总；这些功能仍由 evaluation 管理，不进入具体 backend。
