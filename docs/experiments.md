# 实验指南

以下命令从项目根目录运行，安装与本地模型、数据缓存准备见[项目首页](../README.md#安装与运行前提)。

## 参数计数单位

| 参数 | 单位与范围 |
|---|---|
| `--start-layer-index` / `--max-layers` | 敏感性脚本筛选后的 Linear 模块索引与数量；索引从 0 开始 |
| `--start-block` / `--end-block` | 微调脚本的 Transformer block 编号，左闭右开；省略终点则到模型末尾 |
| `--max-token-blocks` | 微调训练数据的 token block 数量上限 |
| `--sample-start-index` / `--limit` | 敏感性脚本的评测题目起点与样本限制，与模型层选择相互独立 |
| `--eval-limit` | 微调脚本每个评测 task 的样本上限；MMLU 按子任务分别限制 |

## Qwen3 敏感性分析

```bash
# Qwen3 逐层敏感性分析（默认 MMLU，rank 96）
python scripts/run_qwen3_sensitivity.py --limit 1 --max-layers 1
#   --limit 1       每个评测 task 只取 1 个样本（快速验证）
#   --max-layers 1  只分析第 1 个 Linear 模块（配合 --start-layer-index 可分段运行）

# 更换评测 task 时需显式指定指标
python scripts/run_qwen3_sensitivity.py \
  --task gsm8k --metric exact_match_strict_match \
  --limit 1 --max-layers 1
```

敏感性脚本在模型加载前创建实验目录，自动保存 `console.log`（终端输出与异常）、`events.jsonl`（结构化记录）和 `report.md`（报告），终端仍同步显示进度。每次运行使用独立时间戳目录，采用 UTC+8、格式 `YYYYMMDDTHHMMSS`，精确到秒；指定 `--output` 时 `console.log` 和默认 `events.jsonl` 跟随报告目录。后台运行 BoolQ 全量，无需指定日志路径：

```bash
nohup python -u scripts/run_qwen3_sensitivity.py --task boolq --metric acc --num-fewshot 0 --limit none >/dev/null 2>&1 &
```

使用 `--sample-start-index` 按题目分段（从 0 开始），`--limit` 是从该位置取的题数；`--limit none` 表示测到末尾。它与选择 Linear 模块的 `--start-layer-index` 相互独立。例如先评测 HellaSwag 前 8,000 题，再只评测剩余题目：

```bash
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --limit 8000
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --sample-start-index 8000 --limit none
```

非零题目起点仅支持单 task，limit 使用整数或 none，越界起点会报错。分段沿用缓存数据的原始评测顺序；每段分别评测基线和所选 Linear，日志记录题目起点，报告记录实际范围与样本数。两段分别输出结果，暂不自动合并。为了与一次性全量评测使用相同 prompt，分段示例固定 `--num-fewshot 0`；非零 few-shot 的示例抽样可能随分段而变化。

敏感性实验结束后自动生成热力图，横轴为 Transformer 块号，纵轴为七类投影模块。每张图标注 `Test samples: 本次条数 / 评测集总条数`；总数仅指任务使用的 test 或 validation split，group 按叶子任务汇总，不包含训练数据。每个比较指标输出一张 `<报告名>-<指标>-heatmap.png`，与 Markdown 报告保存在同一实验目录，并嵌入报告末尾；默认目录为 `artifacts/sensitivity/<实验名>/<时间戳>/`，使用 `--output` 时跟随该报告路径。准确率和 exact-match 的掉点乘以 100，以百分点（pp）表示；困惑度等使用原始单位。`--heatmap-max` 默认 10，超出色标范围的格子标真实数值，负值表示改善，未评测格显示灰色。绘图依赖可通过 `python -m pip install -e ".[plotting]"` 安装，`[all]` 也包含此依赖。

已有完整实验可直接补图，无需加载模型；从项目根目录运行：

```bash
python scripts/run_qwen3_sensitivity.py \
  --plot-only artifacts/sensitivity/qwen3-mmlu-mpo-rank-96/20260905T182434/events.jsonl
```

## Alpaca 联合压缩与微调

联合压缩从零起始编号 `25`（含）到最后一个 Transformer block 的全部 attention / MLP proj，并用 Alpaca 只微调 MPO 参数：

```bash
python scripts/run_alpaca_finetune.py \
  --start-block 25 --rank 96 \
  --num-train-epochs 1 --batch-size 1 --grad-accum-steps 4
```

脚本依次评测原模型、联合压缩后的模型和微调后的模型，当前评测任务为 MMLU（5-shot，`acc`）和 HellaSwag（10-shot，`acc_norm`）。`--end-block` 指定不包含的结束 block 编号；省略则到模型末尾。每个 block 压缩 attention 的 `q/k/v/o_proj` 和 MLP 的 `gate/up/down_proj`；对于 36 个 block 的 Qwen3-8B，默认压缩编号 25～35 的 77 个投影层，其余参数在微调时冻结。实际层列表保存在 `experiment_config.json` 的 `compressed_layers` 中。

默认使用全部训练文本和全量评测。快速验证完整流程可运行：

```bash
python scripts/run_alpaca_finetune.py \
  --max-token-blocks 1 --num-train-epochs 1 \
  --batch-size 1 --grad-accum-steps 1 \
  --eval-limit 1 --eval-batch-size 1
```

该命令使用一个默认长度为 2048 token 的训练 block，更新一步。文本 tokenize 后追加 EOS 并拼接切块，因此一个 block 不等于一条 Alpaca 原始记录。三阶段评测均限制每个 task 为一条样本，MMLU 按每个子任务各一条计算；few-shot 示例数量保持上述设置。此命令用于验证流程，训练效果和评测稳定性需要更多数据验证。

默认产物目录为 `artifacts/alpaca-finetune/<模型名>_blocks-<起点>-<终点>_rank-<rank>/<YYYYMMDD-HHMMSS>/`，例如 `artifacts/alpaca-finetune/Qwen3-8B_blocks-25-36_rank-96/20260908-143025/`。模型名取模型来源路径的最后一段，终点为实际使用的不包含端点的 block 编号，时间戳使用本地时间。显式传入 `--artifact-root <目录>` 时直接使用该目录。

产物目录包含三阶段结果 `summary.json`、配置 `experiment_config.json`、日志 `experiment.jsonl`、`decompositions/initial/` 和 `decompositions/finetuned/` 下的逐层 artifact，以及 `checkpoints/` 下的训练状态。产物需配合原始模型和相同执行后端使用。断点续训时，在原命令上追加 `--resume-from <checkpoint路径>`，保持模型、数据、层范围、rank、后端及训练配置一致；续训仍会先评测原模型和初始压缩模型，再由训练接口恢复 checkpoint。
