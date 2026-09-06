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

```text
                        scripts/                         实验入口
                           │
                           ▼
                        workflows/                       流程编排
          ┌───────────────┼───────────────┐
          │               │               │
   sensitivity        compress        finetune · inference
          │               │               │
          ▼               ▼               ▼
  ┌───────┴───────┐  ┌────┴────┐  ┌──────┴──────┐
  │  model        │  │ backends│  │ training    │
  │  (加载/替换)   │  │ (分解/  │  │ (训练循环/   │
  │               │  │  构建)  │  │  checkpoint) │
  └───────┬───────┘  └┬───┬───┘   └──────┬──────┘
          │           │   │              │
          │     ┌─────▼───▼──────┐       │
          │     │ representations│  ┌────▼────┐
          │     │ (结构 & artifact)│  │   nn    │
          │     └───────────────┘  │ (PyTorch│
          │                         │  模块)  │
          │                         └────┬────┘
          │                              │
   ┌──────▼──────────────────────────────▼──────┐
   │  evaluation  ·  data  ·  storage  ·  logging  ·  runtime
   │  (评测)      (数据)   (读写)     (日志)      (配置)
   └────────────────────────────────────────────┘
```

## 模块说明

| 模块 | 职责 | 被谁调用 |
|------|------|----------|
| **representations** | 张量网络结构定义（MPO 等）与统一 artifact | backends, evaluation |
| **nn** | 压缩层的 PyTorch 模块接口 | backends, training |
| **backends** | 分解 & 构建适配（native / TensorLy / torchTT / cuTensorNet） | workflows |
| **model** | 加载 Causal LM，列出 / 替换 / 恢复 Linear 层 | workflows |
| **data** | 训练数据加载（HF / JSONL / 本地）与 DataLoader | workflows.training |
| **training** | 通用训练循环、可插拔 loss、checkpoint | workflows.finetune |
| **evaluation** | 压缩指标、lm-eval 评测、推理性能 benchmark | workflows |
| **workflows** | 敏感性、压缩、微调、推理的完整编排 | scripts |
| **storage** | artifact 读写与目录管理 | workflows |
| **logging** | JSON Lines 实验事件记录 | workflows |
| **runtime** | TOML 配置、离线环境 | 全局 |

## 最小示例

```python
import torch
from qcomp import MPOSpec, get_backend

spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
artifact = get_backend("native", "mpo").decompose(torch.randn(4, 4), spec)
linear = get_backend("tensorly", "mpo").build_linear(artifact)
output = linear(torch.randn(3, 4))
```
