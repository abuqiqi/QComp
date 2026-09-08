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
python scripts/run_qwen3_sensitivity.py --limit 1 --max-layers 1
#   --limit 1       每个评测 task 只取 1 个样本（快速验证）
#   --max-layers 1  只分析第 1 层（配合 --start-index 可分段运行）

# 更换评测 task 时需显式指定指标
python scripts/run_qwen3_sensitivity.py \
  --task gsm8k --metric exact_match_strict_match \
  --limit 1 --max-layers 1
```

敏感性脚本在模型加载前创建实验目录，自动保存 `console.log`（终端输出与异常）、`events.jsonl`（结构化记录）和 `report.md`（报告），终端仍同步显示进度。每次运行使用独立时间戳目录；指定 `--output` 时 `console.log` 和默认 `events.jsonl` 跟随报告目录。后台运行 BoolQ 全量，无需指定日志路径：

```bash
nohup python -u scripts/run_qwen3_sensitivity.py --task boolq --metric acc --num-fewshot 0 --limit none >/dev/null 2>&1 &
```

使用 `--sample-start-index` 按题目分段（从 0 开始），`--limit` 是从该位置取的题数；`--limit none` 表示测到末尾。它与选择 Linear 模块的 `--start-index` 相互独立。例如先评测 HellaSwag 前 8,000 题，再只评测剩余题目：

```bash
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --limit 8000
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --sample-start-index 8000 --limit none
```

非零题目起点仅支持单 task，limit 使用整数或 none，越界起点会报错。分段沿用缓存数据的原始评测顺序；每段分别评测基线和所选 Linear，日志记录题目起点，报告记录实际范围与样本数。两段分别输出结果，暂不自动合并。为了与一次性全量评测使用相同 prompt，分段示例固定 `--num-fewshot 0`；非零 few-shot 的示例抽样可能随分段而变化。

敏感性实验结束后自动生成热力图，横轴为 Transformer 块号，纵轴为七类投影模块。每个比较指标输出一张 `<报告名>-<指标>-heatmap.png`，与 Markdown 报告保存在同一实验目录，并嵌入报告末尾；默认目录为 `artifacts/evaluations/<实验名>/<时间戳>/`，使用 `--output` 时跟随该报告路径。准确率和 exact-match 的掉点乘以 100，以百分点（pp）表示；困惑度等使用原始单位。`--heatmap-max` 默认 10，超出色标范围的格子标真实数值，负值表示改善，未评测格显示灰色。绘图依赖可通过 `python -m pip install -e ".[plotting]"` 安装，`[all]` 也包含此依赖。

已有完整实验可直接补图，无需加载模型；从项目根目录运行：

```bash
python scripts/run_qwen3_sensitivity.py \
  --plot-only artifacts/evaluations/qwen3-mmlu-mpo-rank-96/20260905T102434Z/events.jsonl
```

联合压缩从零起始编号 `25`（含）到最后一个 Transformer block 的全部 attention / MLP proj，并用 Alpaca 只微调 MPO 参数：

```bash
python scripts/run_alpaca_finetune.py \
  --start-layer 25 --rank 96 \
  --num-train-epochs 1 --batch-size 1 --grad-accum-steps 4 \
  --artifact-root artifacts/qwen3-layer25-rank96
```

脚本依次评测原模型、联合压缩后的模型和微调后的模型。`--end-layer` 指定不包含的结束 block 编号；省略则到模型末尾。默认使用全部训练文本和全量评测；快速验证可追加 `--max-blocks 8 --eval-limit 1 --eval-batch-size 1`，其中 `max-blocks` 是训练 token block 数量。

产物目录包含三阶段结果 `summary.json`、配置 `experiment_config.json`、日志 `experiment.jsonl`、`decompositions/initial/` 和 `decompositions/finetuned/` 下的逐层 artifact，以及 `checkpoints/` 下的训练状态。产物需配合原始模型和相同执行后端使用。断点续训时，在原命令上追加 `--resume-from <checkpoint路径>`，保持模型、数据、层范围、rank、后端及训练配置一致；续训仍会先评测原模型和初始压缩模型，再由训练接口恢复 checkpoint。

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

| 模块 | 职责 |
|------|------|
| **representations** | 张量网络结构定义（MPO 等）、数学操作与统一 artifact |
| **nn** | 压缩层的 PyTorch 公共接口与基类 |
| **backends** | 分解、构建与具体执行实现（native / TensorLy / torchTT / cuTensorNet；能力因后端而异） |
| **model** | 加载 Causal LM，查找 / 列出 / 替换 / 恢复层 |
| **data** | 数据加载（HF / JSONL / 本地）、文本预处理与 DataLoader 构建 |
| **training** | 通用训练循环、可插拔 loss、checkpoint |
| **evaluation** | 压缩指标、lm-eval 评测、性能 benchmark |
| **workflows** | 敏感性、压缩、微调、推理的流程编排 |
| **storage** | artifact 读写与目录管理 |
| **logging** | JSON Lines 实验事件记录 |
| **runtime** | TOML 配置、离线环境 |

## 最小示例

```python
import torch
from qcomp import MPOSpec, get_backend

spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))
artifact = get_backend("native", "mpo").decompose(torch.randn(4, 4), spec)
linear = get_backend("tensorly", "mpo").build_linear(artifact)
output = linear(torch.randn(3, 4))
```
