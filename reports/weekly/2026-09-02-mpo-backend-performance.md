# Qwen3-8B MPO Backend 性能汇总

## 1. 可用后端

| 配置名 | 开发/维护方 | 实现 | 训练 | 推理 | 特性与限制 |
| --- | --- | --- | --- | --- | --- |
| `native` | 本项目 | 项目自带 PyTorch | 支持 | 支持 | 无额外依赖、数值参考、显存稳定；逐小块 contraction，速度最慢 |
| `tensorly_torch` | TensorLy 开源项目/社区 | TensorLy-Torch 0.5.0 | 支持 | 支持 | 批量执行 MPO 张量收缩，减少 Python 循环和小 GPU 算子；训练和批量打分最快 |
| `torchtt` | Ion Gabriel Ion（个人开源项目） | torchTT 0.4.0 | 支持 | 支持 | 当前速度接近 native |
| `cutensornet` | NVIDIA（cuQuantum SDK） | cuTensorNet 2.13.0 hybrid | 不支持 backward | 支持 | 首次运行会选择并保存高效的 MPO 收缩顺序；长输入可并行处理大量 token，推理较快；单 token 生成回退 native |

开发方信息来源：[TensorLy-Torch 官方仓库](https://github.com/tensorly/torch)、[torchTT 官方 README](https://github.com/ion-g-ion/torchTT/blob/main/README.md)、[NVIDIA cuTensorNet 文档](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/)。

四个后端读取同一种 canonical MPO cores，因此权重可以跨后端加载；optimizer checkpoint
续训仍需使用原训练 backend。

## 2. 固定测试规模

- GPU：单张 A100 80GB。
- 模型：Qwen3-8B；block 18–35 的 108 个 Linear 替换为 rank-96 MPO。
- 训练速度：固定输入长度 512 tokens，batch size 1，gradient accumulation 4；正式计时
  30 个 optimizer step，累计处理 61,440 tokens（相当于 120 个 512-token micro-batch），
  另预热 5 步。模型权重 BF16，MPO 收缩使用 FP32；表中为 30 步中位数。
- 推理速度：本地 HellaSwag 1024 条样本，共 4096 个 loglikelihood request，batch size 8；
  统计 `lm-eval` 的 evaluation time，不含模型加载。模型和 MPO cores 使用 BF16。

## 3. 时间性能

| Backend | 训练一步 median | 训练加速 | HellaSwag-1024 推理 | 推理加速 |
| --- | ---: | ---: | ---: | ---: |
| `native` | 45.698 s | 1.00× | 556.124 s | 1.00× |
| `tensorly_torch` | 2.920 s | 15.65× | 95.191 s | 5.84× |
| `torchtt` | 36.912 s | 1.24× | 461.209 s | 1.21× |
| `cutensornet` | 不支持 | — | 66.243 s | 8.40× |

## 4. 峰值显存

训练显存对应第 2 节的 512-token、gradient accumulation 4 测试：

| Backend | 峰值实际分配 | PyTorch 峰值预留 |
| --- | ---: | ---: |
| `native` | 18.69 GiB | 20.97 GiB |
| `tensorly_torch` | 14.87 GiB | 27.87 GiB |
| `torchtt` | 16.26 GiB | 27.90 GiB |
| `cutensornet` | 不支持训练 | — |

推理显存来自独立的全模型测试：batch size 1、1024-token prefill、32-token decode，
模型和 MPO cores 使用 BF16；它不是 HellaSwag 评测过程的显存统计。

| Backend | 峰值实际分配 | PyTorch 峰值预留 |
| --- | ---: | ---: |
| `native` | 11.79 GiB | 12.04 GiB |
| `tensorly_torch` | 13.80 GiB | 14.83 GiB |
| `torchtt` | 11.79 GiB | 12.14 GiB |
| `cutensornet` | 34.31 GiB | 34.86 GiB |

汇报时以“峰值实际分配”为主；“峰值预留”包含 PyTorch CUDA allocator 缓存。

## 5. 结论

结论：训练优先 `tensorly_torch`，并设置 `training.bf16_autocast: false`。纯推理速度
`cutensornet` 最快；若希望依赖简单、显存较低并更贴近 native 数值结果，则使用
`tensorly_torch`。`torchtt` 保留作兼容性对照。

## 6. 原始结果

- [30-step 训练结果](../../results/benchmarks/qwen3-8b-compactifai-training-alpaca-512-gradacc4-30steps-fp32-tt.json)
- [1024-token native 与 TensorLy-Torch 推理结果](../../results/benchmarks/qwen3-8b-compactifai-inference-backends.json)
- [1024-token torchTT 推理结果](../../results/benchmarks/qwen3-8b-compactifai-torchtt-inference.json)
- [1024-token cuTensorNet 推理结果](../../results/benchmarks/qwen3-8b-compactifai-cutensornet-inference.json)
- [HellaSwag-1024 结果](../../results/evaluations/backend-hellaswag-1024/)
- [训练后 checkpoint 的 HellaSwag-256 交叉验证](../../results/evaluations/backend-hellaswag-256-retrained/)
