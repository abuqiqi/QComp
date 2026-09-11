# qcomp

大模型张量网络压缩工具包——将稠密线性层替换为 MPO 等张量网络结构，并通过敏感性分析、微调和评测完成完整压缩流程。

## 安装与运行前提

使用 Python ≥3.11，从项目根目录安装：

```bash
python -m pip install -e ".[all]"
```

`-e` 为 editable 安装，修改源码后无需重装。`all` 包含模型、数据、评测、绘图及全部后端依赖，包括 CUDA 12 的 cuTensorNet 包。只使用默认 TensorLy 实验脚本可安装 `python -m pip install -e ".[model,data,evaluation,tensorly,plotting]"`；只使用原生 PyTorch 核心接口可安装 `python -m pip install -e .`。

模型加载与实验脚本默认使用 `cuda:0`、BF16，需要可用的 CUDA 设备及足够容纳模型和运行中间数据的显存；CPU 示例需显式指定设备和适用的 dtype。默认运行配置见 [config/runtime.toml](config/runtime.toml)：模型位于 `../../models/Qwen3-8B`，Hugging Face 缓存位于 `../../datasets/huggingface`，均相对于项目根目录解析。

默认 `offline = true`，运行前需要在配置路径准备完整模型、tokenizer，以及所选训练和评测任务的数据缓存。脚本通过 `--model` 覆盖模型来源，或通过 `--runtime-config` 指定另一份配置；允许联网时在该 TOML 中显式设置 `offline = false`。安装 Python 依赖不会自动准备模型和数据。

## 最小示例

以下小矩阵示例在 CPU 上运行，使用 native 分解、TensorLy 执行：

```python
import torch
from qcomp import MPOSpec, get_backend

spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
artifact = get_backend("native", "mpo").decompose(torch.randn(4, 4), spec)
linear = get_backend("tensorly", "mpo").build_linear(artifact)
output = linear(torch.randn(3, 4))
```

## 架构与模块

```text
稠密模型 → 敏感性分析 → 确定压缩方案 → 分解与层替换 → 微调 → 评测与推理
```

workflow 调用分解后端生成统一 artifact，再由执行后端构建压缩层，通过 model 安装到模型中。分解与执行后端可以分别选择。

| 模块 | 职责 |
|---|---|
| [workflows](src/qcomp/workflows/README.md) | 压缩、敏感性分析、微调和推理编排 |
| [representations](src/qcomp/representations/README.md) | 张量网络结构、数学操作和 artifact |
| [nn](src/qcomp/nn/README.md) / [backends](src/qcomp/backends/README.md) | 压缩层公共接口与计算库实现 |
| [data](src/qcomp/data/README.md) / [training](src/qcomp/training/README.md) | 数据准备、训练循环、objective 和 checkpoint |
| [evaluation](src/qcomp/evaluation/README.md) | 压缩指标、lm-eval 和单层性能计时 |
| [源码入口](src/qcomp/README.md) | 模型操作、存储、日志和运行配置 |

新增张量网络表示或训练方法，见[扩展指南](docs/extending.md)。

## 实验入口

- [Qwen3 敏感性分析](docs/experiments.md#qwen3-敏感性分析)：逐 Linear 分解与恢复、题目分段、日志和热力图。
- [交互式敏感性选层](docs/experiments.md#交互式敏感性选层)：离线勾选数据集、调整权重、分析稳定性并导出候选模块。
- [按 JSON 联合压缩与评测](docs/experiments.md#按-json-联合压缩与评测)：读取选层方案、联合压缩、前后评测并保存 artifact。
- [Alpaca 联合压缩与微调](docs/experiments.md#alpaca-联合压缩与微调)：按 Transformer block 选择目标、三阶段评测和断点续训。
