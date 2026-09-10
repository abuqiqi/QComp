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

敏感性实验结束后自动生成热力图，横轴为 Transformer 块号，纵轴为七类投影模块。每张图标注 `Test samples: 本次条数 / 评测集总条数`；总数仅指任务使用的 test 或 validation split，group 按叶子任务汇总，不包含训练数据。图标题同时显示对应指标的 `Baseline` 得分，读取报告的 `Baseline Metrics` 表；准确率和 exact-match 等比例指标显示为百分比，其他指标保留原始单位。每个比较指标输出一张 `<报告名>-<指标>-heatmap.png`，与 Markdown 报告保存在同一实验目录，并嵌入报告末尾；默认目录为 `artifacts/sensitivity/<实验名>/<时间戳>/`，使用 `--output` 时跟随该报告路径。准确率和 exact-match 的掉点乘以 100，以百分点（pp）表示；困惑度等使用原始单位。`--heatmap-max` 默认 10，超出色标范围的格子标真实数值，负值表示改善，未评测格显示灰色。绘图依赖可通过 `python -m pip install -e ".[plotting]"` 安装，`[all]` 也包含此依赖。

已有完整实验可直接补图，无需加载模型；从项目根目录运行：

```bash
python scripts/run_qwen3_sensitivity.py \
  --plot-only artifacts/sensitivity/qwen3-mmlu-mpo-rank-96/20260905T182434/events.jsonl
```

## 交互式敏感性选层

从项目根目录生成可直接打开的独立 HTML，只读取模型配置和敏感度结果，无需加载权重、GPU、网页服务或网络：

```bash
python scripts/build_layer_selection_dashboard.py
# 指定另一份来源配置或新的输出目录
python scripts/build_layer_selection_dashboard.py --config config/layer_selection.json --output artifacts/layer-selection/my-selection
```

默认配置 `config/layer_selection.json` 为五个数据集各指定一份完整结果。相对路径基于配置文件目录；默认使用 BoolQ / MMLU 全量、HellaSwag 合并 2,000 条、GSM8K 合并 256 条和 TriviaQA 128 条。生成器验证完成状态、252 个模块、模型与压缩配置、指标基线及合并来源哈希。合并来源原先位于 `evaluations`、后来移到 `sensitivity` 时，按同任务下的原时间戳目录定位，并严格核对原哈希。缺失或不一致会报错。

默认输出 `artifacts/layer-selection/<北京时间戳>/index.html` 和 `sources.json`，显式输出目录必须尚不存在。将 `index.html` 下载到本机后用浏览器打开即可；页面内已嵌入全部数据和资源。来源配置采用绝对路径保存，便于在同一机器重建页面；换机器时修改来源路径。

页面默认全部勾选、等权、期望选择 32 个矩阵（`requested_module_count`）、rank=96。可切换各任务实际存在的指标，拖动权重即时更新；固定评分为 `Σ 归一化权重 × max(0, 基线−压缩得分) × 100`，不按题数加权。表格保留有符号掉点；可用单任务最大掉点上限过滤候选。热力图与排名表支持点击联动，悬停显示各任务得分。

稳定性模式按 5% 步长枚举上下限内总和为 100% 的权重组合。默认五任务各 10%～30%，共 381 组。勾选数为 N 时，下限按 `50/N%` 向下取整至 5% 倍数，上限按 `150/N%` 向上取整并限制为 100%。点击计算后分批处理，可取消；设置改变后必须重新计算。期望选择数量的边界并列按剩余名额分摊入选频率，分数保留十位小数判定并列；排名统计使用平均名次和 nearest-rank 分位数。

固定模式依次按综合分数、最大掉点、参数收益及模块路径排序；稳定性模式依次按入选频率、最大掉点、最差加权分数、参数收益及模块路径排序。预计参数节省由各模块 `1−1/model_ratio` 求和。A/B 按钮保存当前页面内的完整方案快照，刷新页面会清空；CSV 导出所有模块当前排名及各任务指标。

### 统一 JSON 与计划加载

页面只下载一个 `layer-selection.json`，与未来自动选层共用结构，不保存格式或算法版本号：

- `kind` 为 `qcomp_compression_selection`，`created_at` 和 `producer` 记录生成时间和入口。
- `model` 保存 `name_or_path`、`model_type`、`config_sha256`；配置哈希不代表权重指纹。
- `compression_plan.targets` 按入选顺序保存每个目标的 `module_path`、`representation` 和完整 `spec`。MPO spec 包含 `out_modes`、`in_modes`、`ranks`，允许不同矩阵独立配置。执行以 targets 为唯一依据。
- `selection.method` 为 `fixed_weight` 或 `weight_stability`。`settings` 保存 `requested_module_count`、可空的 `max_task_drop_pp`、`weight_step_pp` 和五任务的勾选、指标、原始权重及上下限；`sources` 保存来源路径、题数、哈希及完整合并 provenance。
- `selection.results` 保存候选总数、合格数量、实际入选数量 `selected_count`、整体参数节省比例、归一化权重，以及全部 252 个矩阵的资格、排名、掉点和参数收益。固定模式的 `stability` 为 null；稳定性模式增加权重组合数、各合格矩阵的入选频率、中位排名、P95 排名和最差加权掉点，不保存逐权重组合的完整排名。
- `evaluation.joint_compression_status` 为 `not_evaluated`，表示尚未进行联合评测。

所有掉点使用百分点（`_pp`），保留原始负掉点；参数节省、频率和归一化权重使用 0～1 比例。不合格候选排名为 null。筛选后实际入选数可以小于期望数量。固定权重模式只保存固定评分；稳定性模式的 `weighted_drop_pp` 是滑块权重下的参考分数，排名由稳定性统计确定。

生成器默认读取日志模型目录的 `config.json`。模型目录迁移后可指定本地配置：

```bash
python scripts/build_layer_selection_dashboard.py --model-config /infini-data/Qwen3-8B/config.json
```

显式 `--model-config` 相对于当前工作目录；生成的 `sources.json` 保存绝对 `model_config` 路径，重建时可直接使用该来源配置。生成器按配置推导七类投影形状，使用与实验脚本共享的 modes 规则，核对每个矩阵的压缩比。当前页面仍只处理 Qwen3 五任务、252 个模块和 rank 96。

页面导入检查模型配置和来源哈希，并重新计算验证完整 targets、设置及结果。稳定性复算显示进度并支持取消；不一致时保留原页面状态。旧文件需通过最新版页面重新导出。当前页面没有对应敏感度数据的混合 rank 方案不能导入复算；Python 计划接口可解析混合 rank 的 MPO 计划。

压缩端将已经加载的模型传入一次调用即可，加载接口自动检查目标存在、无 bias Linear 类型和矩阵维度，不读取敏感度来源、不修改模型：

```python
from qcomp import load_compression_plan

# model 是调用方已加载的模型。
plan = load_compression_plan("layer-selection.json", model=model)
```

返回的 plan 可交给现有 `compress_model()`，backend 和 dtype 继续由执行配置提供。`compression_plan_to_dict(plan)` 与 `compression_plan_from_dict(data)` 分别读写执行部分，供脚本生成统一 JSON；未知表示明确报错，目前仅支持 MPO。模型路径只用于追溯，加载不要求原路径存在，也不宣称校验了模型权重身份。

页面不执行压缩或微调，现有微调脚本暂未增加 JSON 入口。参数节省不代表推理加速，入选频率不代表统计置信度，联合压缩分数仍需实际评测。

验证生成器和计划接口使用 `python -m pytest tests/test_layer_selection_dashboard.py tests/test_compression_plan_io.py`。浏览器交互测试另需安装 `playwright` 和 `python -m playwright install chromium`，运行 `python -m pytest tests/test_layer_selection_browser.py`；未安装 Playwright 时该文件跳过，页面本身没有此依赖。

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

默认产物目录为 `artifacts/alpaca-finetune/<模型名>_blocks-<起点>-<终点>_rank-<rank>/<YYYYMMDDTHHMMSS>/`，例如 `artifacts/alpaca-finetune/Qwen3-8B_blocks-25-36_rank-96/20260908T143025/`。模型名取模型来源路径的最后一段，终点为实际使用的不包含端点的 block 编号，时间戳使用本地时间。显式传入 `--artifact-root <目录>` 时直接使用该目录。

产物目录包含三阶段结果 `summary.json`、配置 `experiment_config.json`、日志 `experiment.jsonl`、`decompositions/initial/` 和 `decompositions/finetuned/` 下的逐层 artifact，以及 `checkpoints/` 下的训练状态。产物需配合原始模型和相同执行后端使用。断点续训时，在原命令上追加 `--resume-from <checkpoint路径>`，保持模型、数据、层范围、rank、后端及训练配置一致；续训仍会先评测原模型和初始压缩模型，再由训练接口恢复 checkpoint。
