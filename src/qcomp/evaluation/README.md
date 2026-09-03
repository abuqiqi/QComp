# Evaluation 代码结构

`evaluation` 负责统一测量压缩结果和计算过程，不实现张量分解或模型层收缩。它接收
representations 层定义的通用 artifact 和 backends 层提供的运行能力，输出可以直接
记录、比较和汇总的指标。

当前实现面向单个无 bias Linear 层。完整模型质量、完整模型性能和 CUDA 显存评测尚未
实现。

## 目录结构

```text
evaluation/
├── __init__.py       # 导出公开评测接口
├── compression.py    # 参数量、压缩率和重建误差
└── performance.py    # 分解、推理和训练 step 计时
```

主要公共对象：

- `CompressionMetrics`：保存参数量、张量字节数、两类压缩率和相对重建误差。
- `compression_metrics`：评测任意已注册张量网络表示的压缩结果。
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

## 完整模型评测

完整模型评测尚未实现。模型层替换、数据集和 workflow 完成后，evaluation 将复用相同
底层测量规则，把被测操作从单层 `linear(inputs)` 扩展为完整模型调用：

```python
outputs = model(**batch)
```

完整模型阶段将增加推理吞吐、训练 step、峰值显存、perplexity、accuracy 和报告汇总；
这些功能仍由 evaluation 管理，不进入具体 backend。
