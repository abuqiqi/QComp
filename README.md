# qcomp

大模型张量网络压缩工具包——将稠密线性层替换为 MPO 等张量网络结构，并通过敏感性分析、微调和评测完成完整压缩流程。

## 压缩流程

```text
稠密模型 → 敏感性分析 → 确定压缩方案 → 张量分解 & 层替换 → 微调 → 评测 & 推理
```

## 安装

```bash
python -m pip install -e ".[all]"
```

> `-e` 为 editable 安装，修改源码后无需重装。运行配置见 `config/runtime.toml`。

## 运行脚本

```bash
# Qwen3 逐层敏感性分析（默认 MMLU，rank 96）
python scripts/run_qwen3_mmlu_sensitivity.py --limit 1 --max-layers 1
#   --limit 1       每个评测 task 只取 1 个样本（快速验证）
#   --max-layers 1  只分析第 1 层（配合 --start-index 可分段运行）

# 更换评测 task 时需显式指定指标
python scripts/run_qwen3_mmlu_sensitivity.py \
  --task gsm8k --metric exact_match_strict_match \
  --limit 1 --max-layers 1
```

## 模块架构

按职责分为四组，下面展示模块分组，不表示调用顺序或依赖关系。

```text
实验入口    scripts

流程编排    workflows
            敏感性分析 · 压缩 · 微调 · 推理

核心能力    representations    张量网络结构与统一 artifact
            nn                 压缩层公共接口
            backends           分解与压缩层实现
            model              模型加载与层替换

配套能力    data · training · evaluation
            数据准备、通用训练与评测
            storage · logging · runtime
            产物读写、日志与运行配置
```

压缩时，workflow 调用分解后端把稠密权重转换为统一 artifact，再由执行后端构建压缩层，最后通过 `model` 安装到模型中。分解后端与执行后端可以分别选择。

## 模块说明

| 模块 | 职责 | 主要调用方 / 使用方 |
|------|------|--------------------|
| **representations** | 张量网络结构定义（MPO 等）、数学操作与统一 artifact | backends, nn, evaluation, storage, workflows, scripts |
| **nn** | 压缩层的 PyTorch 公共接口与基类 | backends, model |
| **backends** | 分解、构建与具体执行实现（native / TensorLy / torchTT / cuTensorNet；能力因后端而异） | workflows.compress, workflows.sensitivity, evaluation, scripts |
| **model** | 加载 Causal LM，查找 / 列出 / 替换 / 恢复层 | workflows.compress, workflows.sensitivity, workflows.finetune, scripts |
| **data** | 数据加载（HF / JSONL / 本地）、文本预处理与 DataLoader 构建 | scripts / 库调用方 |
| **training** | 通用训练循环、可插拔 loss、checkpoint | workflows.finetune / 库调用方 |
| **evaluation** | 压缩指标、lm-eval 评测、性能 benchmark | workflows.sensitivity, scripts / 库调用方 |
| **workflows** | 敏感性、压缩、微调、推理的流程编排 | scripts / 库调用方 |
| **storage** | artifact 读写与目录管理 | workflows.sensitivity, scripts |
| **logging** | JSON Lines 实验事件记录 | workflows.sensitivity, scripts |
| **runtime** | TOML 配置、离线环境 | model, data, evaluation.lm_eval, workflows.sensitivity |

## 最小示例

```python
import torch
from qcomp import MPOSpec, get_backend

spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
artifact = get_backend("native", "mpo").decompose(torch.randn(4, 4), spec)
linear = get_backend("tensorly", "mpo").build_linear(artifact)
output = linear(torch.randn(3, 4))
```
