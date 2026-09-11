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

敏感性脚本在模型加载前创建实验目录，自动保存 `console.log`（终端输出与异常）、`sensitivity_results.json`（完整结构化结果）和 `report.md`（报告），终端仍同步显示进度。每次运行使用独立时间戳目录，采用 UTC+8、格式 `YYYYMMDDTHHMMSS`，精确到秒；指定 `--output` 时 `console.log` 和默认 `sensitivity_results.json` 跟随报告目录。后台运行 BoolQ 全量，无需指定日志路径：

```bash
nohup python -u scripts/run_qwen3_sensitivity.py --task boolq --metric acc --num-fewshot 0 --limit none >/dev/null 2>&1 &
```

使用 `--sample-start-index` 按题目分段（从 0 开始），`--limit` 是从该位置取的题数；`--limit none` 表示测到末尾。它与选择 Linear 模块的 `--start-layer-index` 相互独立。例如先评测 HellaSwag 前 8,000 题，再只评测剩余题目：

```bash
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --limit 8000
python scripts/run_qwen3_sensitivity.py --task hellaswag --metric acc_norm --num-fewshot 0 --sample-start-index 8000 --limit none
```

非零题目起点仅支持单 task，limit 使用整数或 none，越界起点会报错。分段沿用缓存数据的原始评测顺序；每段分别评测基线和所选 Linear，结果记录题目起点，报告记录实际范围与样本数。两段分别输出结果，暂不自动合并。为了与一次性全量评测使用相同 prompt，分段示例固定 `--num-fewshot 0`；非零 few-shot 的示例抽样可能随分段而变化。

敏感性实验结束后自动生成热力图，横轴为 Transformer 块号，纵轴为七类投影模块。每张图标注 `Test samples: 本次条数 / 评测集总条数`；总数仅指任务使用的 test 或 validation split，group 按叶子任务汇总，不包含训练数据。图标题同时显示对应指标的 `Baseline` 得分，读取报告的 `Baseline Metrics` 表；准确率和 exact-match 等比例指标显示为百分比，其他指标保留原始单位。每个比较指标输出一张 `<报告名>-<指标>-heatmap.png`，与 Markdown 报告保存在同一实验目录，并嵌入报告末尾；默认目录为 `artifacts/sensitivity/<实验名>/<时间戳>/`，使用 `--output` 时跟随该报告路径。准确率和 exact-match 的掉点乘以 100，以百分点（pp）表示；困惑度等使用原始单位。`--heatmap-max` 默认 10，超出色标范围的格子标真实数值，负值表示改善，未评测格显示灰色。绘图依赖可通过 `python -m pip install -e ".[plotting]"` 安装，`[all]` 也包含此依赖。

已有完整实验可直接补图，无需加载模型；从项目根目录运行：

```bash
python scripts/run_qwen3_sensitivity.py \
  --plot-only "artifacts/sensitivity/qwen3-mmlu-mpo-rank-96/<运行时间戳>/sensitivity_results.json"
```

### 标准敏感性结果

实验在基线完成及每个 case 完成后原子更新 `sensitivity_results.json`。文件包含运行状态、预期模块名单、模型配置快照及哈希、实际模型参数量、执行设置、完整 `EvaluationTaskConfig`、实际 `EvaluationTask`、基线和逐层结果。每个 case 保存 `compression_plan.targets`、模块及模型压缩统计、得分、退化、误差和耗时。

状态为 `running`、`completed`、`failed` 或 `interrupted`；强制终止后保留最后一次成功保存的进度。完成结果可直接被页面读取，不支持自动续跑。`--results` 可指定结果文件，默认与报告同目录；已有结果不会覆盖。`SensitivityExperimentResult.results_path` 返回结果位置。报告或绘图失败不会改变已完成的评测状态。

```python
from qcomp.workflows import read_sensitivity_results

result = read_sensitivity_results("sensitivity_results.json")
print(result["evaluation_config"], result["cases"][0]["compression_plan"])
```

### 历史结果转换

```bash
python scripts/migrate_sensitivity_results.py
```

转换器默认扫描 `artifacts/sensitivity`，将已完成的历史 JSONL 与合并结果写入 `migrated-<北京时间戳>/`，内部保留原实验目录层级。它校验原来源哈希、合并关系、题数和基线，按历史 Qwen3 规则重建并核对 Spec；不运行评测。原文件保持不变，无法恢复的信息标记为未知，保留 few-shot 默认设置与分段说明。

输出 `migration_manifest.json` 列出成功、跳过和失败及原文件哈希；未完成实验跳过，完整实验验证失败时返回非零。若当前页面配置的所有来源均转换成功，额外生成指向新结果的 `layer_selection.json`。可用 `--root`、`--output`、`--config` 指定扫描目录、新输出目录和原来源配置。

## 交互式敏感性选层

从项目根目录生成可直接打开的独立 HTML，只读取模型配置和敏感度结果，无需加载权重、GPU、网页服务或网络：

```bash
python scripts/build_layer_selection_dashboard.py
# 指定另一份来源配置或新的输出目录
python scripts/build_layer_selection_dashboard.py --config config/layer_selection.json --output artifacts/layer-selection/my-selection
```

来源配置使用非空 `datasets` 数组，每项包含唯一 `id`、显示名称 `label`、默认指标 `default_metric` 和指向 `sensitivity_results.json` 的 `path`；相对路径基于配置文件目录。任务、指标方向、题数、基线、完整 MPO Spec 和模型配置由结果文件提供，不加载模型、数据集或原报告。`id` 可以与实际 task 不同，从而区分同任务的不同评测设置。

```json
{
  "datasets": [
    {"id": "boolq", "label": "BoolQ", "default_metric": "acc", "path": "../results/boolq/sensitivity_results.json"}
  ]
}
```

生成器要求各来源已完成，模型身份和执行配置一致，覆盖相同模块集合，且对应模块的完整 Spec、参数量和模型压缩比一致。不同模块允许不同 modes、核数和逐 bond ranks；每个模块只有一个已评测 Spec。缺少模块或 Spec 不同会明确报错。

默认输出 `artifacts/layer-selection/<北京时间戳>/index.html` 和 `sources.json`，显式输出目录必须尚不存在。页面嵌入全部结果及资源，可直接离线打开。来源配置保存绝对结果路径，换机器重建时只需调整结果路径。

页面默认全部数据集勾选、等权，期望选择 `min(32, 模块数)` 个矩阵。支持 `higher` 和 `lower` 的 0～1 比例指标：前者退化为 `基线−压缩得分`，后者为 `压缩得分−基线`，乘以 100 转为百分点。综合分数为 `Σ 归一化权重 × max(0, 退化百分点)`，不按题数加权；表格保留负退化，可按最大单任务退化过滤。

稳定性模式按 5% 步长枚举上下限内总和为 100% 的权重组合。勾选数为 N 时，下限按 `50/N%` 向下取整至 5% 倍数，上限按 `150/N%` 向上取整并限制为 100%；单任务固定为 100%。五任务默认各 10%～30%，共 381 组。枚举前检查组合数，超过 100,000 组需收窄范围或使用固定权重。计算可取消，设置改变后重新计算。边界并列按剩余名额分摊入选频率，分数保留十位小数判定并列，排名采用平均名次和 nearest-rank 分位数。

固定模式依次按综合分数、最大退化、参数收益和模块路径排序；稳定性模式依次按入选频率、最大退化、最差加权退化、参数收益和路径排序。模块压缩比由 `MPOSpec` 的参数量属性计算并核对实际统计。新实验按节省参数量除以原模型总参数量计算收益；缺少总参数量的历史转换结果使用 `1−1/model_ratio` 汇总，页面和导出注明依据。

模型标题、任务卡片、表格和模块数量动态生成。热力图根据模块路径中的块号及剩余路径分组；无法分组时提供模块列表。详情显示完整 Spec、参数量和压缩比。A/B 快照只在当前页面保留，CSV 导出所有模块的排名与各任务指标。

### 统一 JSON 与计划加载

页面下载的 JSON 文件名包含模型、压缩格式、最大秩、选层数和时间戳（如 `qwen3-8b-mpo-rank96-32layers-20260911T143923.json`），与未来自动选层共用结构，不保存格式或算法版本号：

- `kind` 为 `qcomp_compression_selection`，`created_at` 和 `producer` 记录生成时间和入口。
- `model` 保存 `name_or_path`、`model_type`、`config_sha256`；配置哈希不代表权重指纹。
- `compression_plan.targets` 按入选顺序保存每个目标的 `module_path`、`representation` 和完整 `spec`。MPO spec 包含 `out_modes`、`in_modes`、`ranks`，允许不同矩阵独立配置。执行以 targets 为唯一依据。
- `selection.method` 为 `fixed_weight` 或 `weight_stability`。`settings` 保存 `requested_module_count`、可空的 `max_task_drop_pp`、`weight_step_pp` 和各任务的勾选、指标、原始权重及上下限；`sources` 保存来源路径、题数、哈希、评测配置和完整 provenance。
- `selection.results` 保存候选总数、合格数量、实际入选数量 `selected_count`、整体参数节省比例、归一化权重，以及全部候选矩阵的资格、排名、掉点和参数收益。固定模式的 `stability` 为 null；稳定性模式增加权重组合数、各合格矩阵的入选频率、中位排名、P95 排名和最差加权掉点，不保存逐权重组合的完整排名。
- `evaluation.joint_compression_status` 为 `not_evaluated`，表示尚未进行联合评测。

所有掉点使用百分点（`_pp`），保留原始负掉点；参数节省、频率和归一化权重使用 0～1 比例。不合格候选排名为 null。筛选后实际入选数可以小于期望数量。固定权重模式只保存固定评分；稳定性模式的 `weighted_drop_pp` 是滑块权重下的参考分数，排名由稳定性统计确定。

页面导入按当前模型配置、来源哈希及完整 Spec 验证并复算方案。稳定性复算显示进度且支持取消；不一致时保留原页面状态。页面只能对具有对应敏感性结果的目标与 Spec 复算，不能修改 Spec 后沿用原得分。

压缩端将已经加载的模型传入一次调用即可，加载接口自动检查目标存在、无 bias Linear 类型和矩阵维度，不读取敏感度来源、不修改模型：

```python
from qcomp import load_compression_plan

# model 是调用方已加载的模型。
plan = load_compression_plan("layer-selection.json", model=model)
```

返回的 plan 可交给现有 `compress_model()`，backend 和 dtype 继续由执行配置提供。`compression_plan_to_dict(plan)` 与 `compression_plan_from_dict(data)` 分别读写执行部分，供脚本生成统一 JSON；未知表示明确报错，目前仅支持 MPO。模型路径只用于追溯，加载不要求原路径存在，也不宣称校验了模型权重身份。

页面不执行压缩或微调，现有微调脚本暂未增加 JSON 入口。参数节省不代表推理加速，入选频率不代表统计置信度，联合压缩分数仍需实际评测。

验证生成器和计划接口使用 `python -m pytest tests/test_layer_selection_dashboard.py tests/test_compression_plan_io.py`。浏览器交互测试另需安装 `playwright` 和 `python -m playwright install chromium`，运行 `python -m pytest tests/test_layer_selection_browser.py`；未安装 Playwright 时该文件跳过，页面本身没有此依赖。

## 按 JSON 联合压缩与评测

`run_compression_plan.py` 读取页面导出的完整方案，通过 `evaluate_compression_plans()` 执行实验，按 `compression_plan.targets` 指定的矩阵、MPO modes 和逐层 ranks 联合压缩，不微调。模型来源优先使用 `--model`，其次为 JSON 的 `model.name_or_path`，最后为 runtime 配置。默认 TensorLy 分解和执行、CUDA 0、BF16 模型、FP32 分解；分解结果转回原层精度。

从项目根目录先检查配置，再运行：

```bash
python scripts/run_compression_plan.py \
  --selection-json artifacts/layer-selection/20260910T215301/layer-selection.json \
  --dry-run

python scripts/run_compression_plan.py \
  --selection-json artifacts/layer-selection/20260910T215301/layer-selection.json
```

`--dry-run` 只解析文件和展示实际执行配置，不加载模型、不创建产物目录；实际运行加载模型后由 `load_compression_plan()` 自动检查目标类型和维度。输入 JSON 原样复制并记录 SHA-256，压缩目标不会按评分再次筛选。

默认评测 JSON 中勾选的数据集，使用 `selection.sources[].provenance.evaluation_config` 中的 few-shot、limit、样本起点、seed、chat template、batch size 和上下文长度。原模型与联合压缩模型使用同一组 evaluator；基线重新评测，不将历史单矩阵分数当作本次基线。`num_fewshot: null` 表示沿用当前安装的 lm-eval 任务默认值，零保持为零。无需读取原敏感度日志，但评测需要本地模型及任务数据缓存。

- `--eval-config config/compression_evaluation.json`：使用独立多任务配置，完整替代选层 JSON 的勾选任务。
- `--no-plot`：不生成得分图；默认评测完成后输出 `scores.png`。
- `--skip-eval`：只压缩并保存，不需要 JSON 内的分析来源；报告明确标记未评测。
- `--eval-limit 16`：临时覆盖每个任务的评测样本上限，保持原样本起点。对于 MMLU 等 group，该值是每个子任务的上限。
- `--eval-batch-size 4`：覆盖评测 batch size。
- `--model /path/to/model`、`--runtime-config ...`：调整模型或运行环境。
- `--decomposition-provider`、`--execution-provider`、`--model-dtype`、`--decomposition-dtype`：控制执行后端与精度。
- `--output ...`：指定新的产物目录，已有目录拒绝覆盖。

独立配置示例见 `config/compression_evaluation.json`。每项必填 `task` 和非空 `metrics`；其他评测参数使用 `LMEvalConfig` 默认值，可明确指定不同任务的 few-shot、题数和样本起点。指标方向自动查询注册表，未知指标可提供完整 `metric_directions` 映射。

```bash
python scripts/run_compression_plan.py \
  --selection-json artifacts/layer-selection/20260910T215301/layer-selection.json \
  --eval-config config/compression_evaluation.json \
  --dry-run
```

`--eval-limit` 和 `--eval-batch-size` 最后覆盖所有任务；`--skip-eval` 不能与 `--eval-config` 或评测覆盖参数同时使用。`summary.json` 的 `comparison` 按任务、指标两层组织，支持同一任务多个指标。任务列表来自独立配置时，不与选层 JSON 合并。库接口支持多个计划；本脚本一次读取一个计划文件。

后台运行：

```bash
nohup python -u scripts/run_compression_plan.py \
  --selection-json artifacts/layer-selection/20260910T215301/layer-selection.json \
  > compression-start.log 2>&1 &
```

启动日志位于当前目录的 `compression-start.log`。默认产物位于 `artifacts/compression/<模型名>/<北京时间戳>/`，包含：

| 文件 | 内容 |
|---|---|
| `selection.json` | 输入方案的原样副本 |
| `experiment_config.json` | 实际模型、后端、精度、种子、评测配置及输入哈希 |
| `decompositions/0000.pt` 等 | 按目标顺序保存的逐矩阵 artifact，保留各自 spec |
| `artifacts.json` | 模块路径与相对 artifact 路径的对应关系 |
| `events.jsonl`、`console.log` | 结构化事件、终端输出和异常 |
| `summary.json`、`report.md` | 实测参数收益、两阶段得分、实际题数、退化和耗时 |
| `scores.png` | 每个任务、每个关注指标独立子图，标注前后分数与指标方向 |

退化按指标方向计算，正值表示变差，报告使用指标原单位。参数节省不代表推理加速。保存的是张量网络 artifact，不是独立的 Hugging Face 模型目录；后续使用需加载原模型，再通过 `load_artifact()`、执行后端的 `build_linear()` 和 `replace_linear()` 安装已保存的压缩层。

联合评测前先保存 artifact，评测失败仍保留已写入产物并记录 `experiment_failed`。JSON 和 Markdown 在绘图前保存，绘图失败仍保留实测结果，失败事件记录 `stage=plotting`，不会记录整个实验成功。默认绘图依赖 Matplotlib，模型加载前检查；可用 `--no-plot` 禁用。脚本退出前恢复内存中的原始 Linear。测试使用小模型实际分解与前向，不运行真实语言模型：

```bash
python -m pytest tests/test_evaluate.py tests/test_compression_plan_script.py tests/test_compression_plan_io.py
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

默认产物目录为 `artifacts/alpaca-finetune/<模型名>_blocks-<起点>-<终点>_rank-<rank>/<YYYYMMDDTHHMMSS>/`，例如 `artifacts/alpaca-finetune/Qwen3-8B_blocks-25-36_rank-96/20260908T143025/`。模型名取模型来源路径的最后一段，终点为实际使用的不包含端点的 block 编号，时间戳使用本地时间。显式传入 `--artifact-root <目录>` 时直接使用该目录。

产物目录包含三阶段结果 `summary.json`、配置 `experiment_config.json`、日志 `experiment.jsonl`、`decompositions/initial/` 和 `decompositions/finetuned/` 下的逐层 artifact，以及 `checkpoints/` 下的训练状态。产物需配合原始模型和相同执行后端使用。断点续训时，在原命令上追加 `--resume-from <checkpoint路径>`，保持模型、数据、层范围、rank、后端及训练配置一致；续训仍会先评测原模型和初始压缩模型，再由训练接口恢复 checkpoint。

### 压缩与评测的种子

敏感度入口使用 `--evaluation-seed 42`，联合压缩入口使用 `--decomposition-seed 42`。两个入口仍接受 `--seed`，分别保持各自的评测或分解语义；不能同时传入新旧参数。敏感度没有独立分解种子，分解沿用评测结束后的随机状态。联合压缩的分解种子保留模型加载前及分解前的设置时机，任务评测种子独立来自评测配置。

独立评测 JSON 使用 `evaluation_seed`，默认 42；也接受旧 `seed`，但两者不能同时出现。历史敏感度事件及选层来源中的 `seed` 字段保持不变，在读取时转换。新联合压缩 `experiment_config.json` 保存 `decomposition_seed`，每个任务的执行配置保存 `evaluation_seed`，包括解析后的默认值与 CLI 覆盖结果。
