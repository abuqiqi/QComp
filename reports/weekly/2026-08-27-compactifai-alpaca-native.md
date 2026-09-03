# Qwen3-8B CompactifAI 风格压缩 + Alpaca Healing 实验

这轮实验已经收尾。Qwen3-8B 后 18 个 block 的 108 个投影矩阵被替换为 D=96 的三节点 MPO，完整模型参数量下降 28.67%。Alpaca healing 跑了 600 steps，训练和评测链路都正常，但五项下游任务表明模型能力远未恢复。

## 实验目的

目标是在单张 A100 80GB 上完成一次有实际规模的 TT 压缩实验：压缩 Qwen3-8B 的中后期层，用本地 Alpaca 数据只训练 TT cores，并打通分解、续训、保存、重载和评测流程。

这轮先解决工程问题，再用全量评测确认这种压缩强度是否可用。所有配置、checkpoint 和评测结果都保留在仓库中。

## 压缩方案

当前配置主要参考 [CompactifAI 论文](https://arxiv.org/abs/2401.14109) 的压缩流程和层敏感性分析，不是从 Qwen3-8B 上重新搜索出来的最优方案。论文中的依据与本实验的对应关系如下。

| CompactifAI 中的做法或观察 | 本实验的处理 |
|---|---|
| 将 decoder block 中的 Self Attention 和 MLP 权重矩阵分解为 MPO | 压缩每个目标 block 的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj` |
| 先做层敏感性分析；LLaMA-2 7B 的前段 block 更敏感，中段到后段更适合大幅压缩 | 只压缩 Qwen3-8B 的后半段 block 18-35，block 0-17 保持 Dense |
| 每个 block 最后的 MLP 输出层更敏感，论文建议不做 tensorization | 在 Qwen3-8B 中把对应的 MLP 输出投影视为 `down_proj`，因此所有 `down_proj` 保持 Dense |
| bond dimension 控制压缩率与精度；论文的 LLaMA-2 7B 基准使用约 100 的 bond dimension | 采用接近该量级、同时适配当前三节点分解的 `D_max=96` |
| 逐层 SVD 截断没有考虑层间影响，因此压缩后要做短程 healing；论文使用 UltraChat、Alpaca 和 OpenHermes，训练少于一轮 | 使用本地 Alpaca，跑 600 optimizer steps，约覆盖 0.23 个 epoch |

这里有两个属于本实验自己的取舍。第一，论文的敏感性结论来自 LLaMA-2 7B，我们直接迁移到 Qwen3-8B，没有先做逐层 profiling。第二，`D_max=96` 只是参考论文约 100 的量级后选定的工程值，不是论文给出的 Qwen 配置，也没有经过 rank sweep。

## 压缩规模

Dense 权重形状按 PyTorch Linear 的 `[out_features, in_features]` 记录。每个 MPO 有三个 core，canonical core 布局为 `[r_left, out_mode, in_mode, r_right]`。

| 投影 | Dense 权重形状 | MPO out modes | MPO in modes | 三个 MPO core 的形状 | TT ranks | Dense 参数/矩阵 | TT 参数/矩阵 | 压缩倍数 | 参数减少 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| `q_proj/o_proj` | `[4096, 4096]` | `[16, 16, 16]` | `[16, 16, 16]` | `[1,16,16,96]` / `[96,16,16,96]` / `[96,16,16,1]` | `[1, 96, 96, 1]` | 16.78M | 2.41M | 6.97× | 85.64% |
| `k_proj/v_proj` | `[1024, 4096]` | `[8, 8, 16]` | `[16, 16, 16]` | `[1,8,16,96]` / `[96,8,16,96]` / `[96,16,16,1]` | `[1, 96, 96, 1]` | 4.19M | 1.22M | 3.45× | 71.00% |
| `gate_proj/up_proj` | `[12288, 4096]` | `[16, 16, 48]` | `[16, 16, 16]` | `[1,16,16,96]` / `[96,16,16,96]` / `[96,48,16,1]` | `[1, 96, 96, 1]` | 50.33M | 2.46M | 20.48× | 95.12% |

| 汇总范围 | Dense 参数 | TT/压缩后参数 | 压缩倍数 | 参数减少 |
|---|---:|---:|---:|---:|
| 每个目标 block 的 6 个矩阵 | 142.61M | 12.17M | 11.72× | 91.47% |
| 18 个 block 的 108 个目标矩阵 | 2.567B | 218.97M | 11.72× | 91.47% |
| 完整模型 | 8.191B | 5.843B | 1.40× | 28.67% |

## Healing 数据和训练设置

Alpaca 从本地 Hugging Face 缓存加载：`tatsu-lab/alpaca` 的 `train` split，共 52,002 条记录，直接使用已有 `text` 字段。数据总计约 5,386,125 tokens，按长度 512 连续打包为 10,520 个 blocks。

- 训练 600 optimizer steps，batch size 1，gradient accumulation 4。GPU 每次处理 1 条 512-token 序列；连续处理 4 条后更新一次 TT cores。总共更新 600 次，因此实际处理 2,400 条序列、1,228,800 tokens。
- 实际处理 1,228,800 tokens，约是完整一轮数据的 22.81%。
- 学习率 `1e-5`，BF16 autocast；模型和 TT activation checkpointing 都开启。
- 只训练约 218.97M 个 FP32 TT core 参数。Embedding、LM head、block 0–17、全部 `down_proj`、normalization 等 Dense 参数全部冻结。
- 每 150 steps 保存一次，最多保留两个 checkpoint；中断后只接受训练配置和数据签名一致的最新 checkpoint。
- 训练主流程不内嵌下游评测，也不把 loss 下降直接解释成模型能力恢复；训练完成后另行执行五任务全量评测。

成功主流程用了 9 小时 37 分 36 秒，其中 optimizer 训练 9 小时 35 分 42 秒，其余时间用于分解产物校验、数据准备、最终保存和离线重载验证。运行硬件为单张 NVIDIA A100 80GB，训练峰值显存 23.31 GiB。
正式训练的第 1 个 optimizer step 同时作为 smoke gate：只有 loss 为有限值且训练峰值显存低于 80 GiB，脚本才会继续后面的 599 steps。

## 复现实验

在项目根目录执行：

```bash
bash scripts/run_compactifai_alpaca_native.sh start
bash scripts/run_compactifai_alpaca_native.sh status
bash scripts/run_compactifai_alpaca_native.sh tail
```

`start` 会用 `nohup` 后台启动并立即返回。关闭聊天窗口或本地电脑不会影响远程开发机上的进程，但如果平台把远程开发机关停，训练也会停止；重新开机后再次运行 `start` 会从签名匹配的最新 checkpoint 续训。

配置文件：[压缩配置](../../configs/compression/qwen3_8b_compactifai_d96_blocks18_35.json)；[训练配置](../../configs/experiments/qwen3_8b_compactifai_d96_blocks18_35_alpaca_600.json)。

## Quick run（正式实验前链路验证）

2026-08-28 在正式 108 矩阵实验前，使用真实 Qwen3-8B、真实本地 Alpaca 和 native TT backend 做了一个隔离的 quick run。选取 block 35，把正式方案中每种目标矩阵各替换一个：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj`，共 6 个矩阵；`down_proj` 仍保持 Dense。TT 结构与正式实验相同（3 节点、D=96），训练 1 个 optimizer step、序列长度 128。

| 检查项 | Quick run 实际结果 |
|---|---|
| 覆盖的矩阵类型 | `q/k/v/o/gate/up` 六类各 1 个，均位于 block 35 |
| 目标矩阵参数量 | 142,606,336 → 12,165,120，压缩 11.72×，减少 91.47% |
| TT-SVD | 六个真实 Qwen 权重均分解成功；`q_proj` 命中已有缓存，其余五个约 20 秒完成 |
| 初始重构误差 | relative L2 为 0.7980–0.9679；D=96 很激进，正式训练存在较高的能力损失风险 |
| Alpaca healing | 成功完成 1 step，实际处理 128 tokens |
| Loss | 2.5297，有限值 |
| 可训练参数 | 12,165,120，全部为 native FP32 TT cores；其余 Dense 参数冻结 |
| 训练峰值显存 | 16.53 GiB |
| 六类权重是否真的更新 | 是；12,165,120 个 core 数值全部发生变化，最大变化约 1.0e-5 |
| Checkpoint / final | `checkpoint-1` 和 BF16 `final/index.json` 均存在且哈希可加载 |
| 离线重载生成 | 六个 TT 模块均成功重载并完成生成；峰值显存 15.28 GiB |
| 固定 prompt 输出 | 模型返回 16 个乱码替代符；说明链路可用，但 1 step 不足以修复激进分解造成的能力损失 |

Quick run 还完成了四项正式运行前检查：确认 `decompose/finetune` 可通过 `python -m` 启动；确认 EXIT trap 能记录当前阶段；在 Torch 2.5.1 下先初始化 CUDA 再采集显存；计算重构误差前把缓存 cores 与原始权重放到同一设备。

可追溯记录：[六矩阵分解配置](../../configs/smoke/qwen3_8b_compactifai_quick_block35_decompose.json)、[六矩阵训练配置](../../configs/smoke/qwen3_8b_compactifai_quick_block35_alpaca_1step.json)、[分解结果](../../results/smoke/compactifai-quick-block35/decomposition-result.json)、[训练 run.json](../../artifacts/smoke/compactifai-quick-block35/finetuned/run.json)、[离线重载结果](../../results/smoke/compactifai-quick-block35/offline-verification.json)。

Quick run 跑通了六类矩阵的真实权重分解、TT-only 反向传播、checkpoint、final 导出和离线重载。它也提前暴露了 D=96 重构误差过高的问题；后面的正式评测证明 600-step healing 确实不够。

<!-- AUTO_RESULTS_START -->
## 实验结果（脚本自动更新）

**当前状态：成功。** 当前阶段：全部完成。

| 项目 | 实际结果 |
|---|---|
| 开始 / 结束 | 2026-08-28T01:44:42+00:00 / 2026-08-28T11:22:18+00:00 |
| 成功主流程耗时 | 9 小时 37 分 36 秒 |
| optimizer steps / 实际训练 tokens | 600 / 1,228,800 |
| 观测首段 / 末段 / 最低 loss | 11.3427 / 2.5633 / 2.3841 |
| loss 变化（末段－首段） | -8.7794；负数表示训练日志中的 loss 总体下降 |
| 峰值 GPU 显存 | 23.31 GiB |
| 保留的 checkpoint | 2 个；最后一个：[checkpoint](../../artifacts/finetuned/qwen3-8b-compactifai-d96-blocks18-35-alpaca-600/checkpoint-600) |
| 最终 TT artifact | [final/index.json](../../artifacts/finetuned/qwen3-8b-compactifai-d96-blocks18-35-alpaca-600/final/index.json) |
| 目标矩阵参数量 | 2,566,914,048 → 218,972,160，约 11.72× |
| 完整模型参数量 | 8,190,735,360 → 5,842,793,472，减少 28.67% |
| 最终模型离线重载 | 成功 |
| 原始训练记录 | [run.json](../../artifacts/finetuned/qwen3-8b-compactifai-d96-blocks18-35-alpaca-600/run.json) |
| 原始分解记录 / 日志 | [decomposition result](../../results/runs/compactifai-alpaca-native/decomposition-result.json) / [日志](../../results/runs/compactifai-alpaca-native/run.log) |

固定 prompt 的生成结果：

> 请用两句话解释张量网络压缩。### Input: 1. 1-2-3-4 - 1

<!-- AUTO_RESULTS_END -->

## 正式五任务全量评测

2026-08-28 至 2026-08-30，在同一张 NVIDIA A100 80GB 上使用 lm-eval==0.4.12 串行评测三种模型：原始 Dense Qwen3-8B、未经 healing 的 D=96 MPO、完成 600 steps Alpaca healing 的 MPO。MMLU 和 GSM8K 使用 5-shot，其余任务使用 0-shot；所有任务均为全量数据（limit: null），batch size 8、最大上下文长度 4096、seed 42、不应用 chat template。

评测配置见[五任务三模型配置](../../configs/experiments/qwen3_8b_compactifai_standard_eval.json)，机器可读汇总见[summary.json](../../results/evaluations/compactifai-standard/summary.json)，原始分项结果和详细对比见[comparison.md](../../results/evaluations/compactifai-standard/comparison.md)。

### 能力结果

分数和差值均以百分点表示；恢复率表示 healing 恢复了多少“Dense 到原始 MPO 的能力损失”。

| 任务 / 指标 | Dense | 原始 MPO | Healing 后 MPO | 原始 MPO 相对 Dense | Healing 后相对 Dense | Healing 增益 | 恢复率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMLU 5-shot / accuracy | 74.80 | 25.23 | 23.42 | -49.57 | -51.38 | -1.82 | -3.7% |
| HellaSwag 0-shot / normalized accuracy | 74.96 | 30.19 | 35.48 | -44.76 | -39.47 | +5.29 | 11.8% |
| BoolQ 0-shot / accuracy | 86.73 | 40.70 | 61.53 | -46.02 | -25.20 | +20.83 | 45.2% |
| TriviaQA 0-shot / exact match | 32.03 | 0.00 | 0.20 | -32.03 | -31.83 | +0.20 | 0.6% |
| GSM8K 5-shot / strict exact match | 87.87 | 0.00 | 0.00 | -87.87 | -87.87 | +0.00 | 0.0% |

Healing 对 BoolQ 和 HellaSwag 有改善，但 MMLU 略降，TriviaQA 几乎没有恢复，GSM8K 仍为 0。loss 下降说明 TT cores 能正常训练，不代表通用能力已经回来。对 108 个矩阵统一使用 D=96 过于激进，600-step Alpaca TT-only healing 没有补回分解损失。

### 实际运行时间

下表按成功评测运行的连续完成时间戳统计，包含每个分项的模型加载、安装 TT 模块和评测开销。任务串行执行，因此同一列分项耗时之和等于该模型五任务总耗时。

| 任务 | Dense | 原始 MPO | Healing 后 MPO |
|---|---:|---:|---:|
| MMLU 5-shot | 19 分 00 秒 | 8 小时 57 分 27 秒 | 8 小时 57 分 40 秒 |
| HellaSwag 0-shot | 6 分 32 秒 | 2 小时 44 分 00 秒 | 2 小时 44 分 07 秒 |
| BoolQ 0-shot | 1 分 04 秒 | 25 分 36 秒 | 25 分 37 秒 |
| TriviaQA 0-shot | 2 小时 39 分 57 秒 | 9 小时 01 分 02 秒 | 6 小时 56 分 48 秒 |
| GSM8K 5-shot | 25 分 40 秒 | 1 小时 55 分 32 秒 | 1 小时 54 分 58 秒 |
| **五任务合计** | **3 小时 32 分 13 秒** | **23 小时 03 分 37 秒** | **20 小时 59 分 10 秒** |

成功运行从 2026-08-28T16:11:28+00:00 到 2026-08-30T15:46:28+00:00，三种模型共 15 项评测的总墙钟时间为 **47 小时 35 分 00 秒**。因此只重跑最终 Healing 模型应预留约 21–24 小时，重跑完整三方对比应预留约 48 小时。TT 版本显著慢于 Dense，说明参数量下降没有转化为当前 native backend 上的同比推理加速。原始时间戳见[评测日志](../../results/evaluations/compactifai-standard/run.log)。

### 全流程时间总览

| 阶段 | 实际时间 |
|---|---:|
| 分解产物校验、数据准备、训练、保存和离线验证 | 9 小时 37 分 36 秒 |
| 其中 optimizer 训练 | 9 小时 35 分 42 秒 |
| 最终 108 个 TT 模块离线重载与短生成 | 8.89 秒 |
| Dense / MPO / Healing 后 MPO 五任务全量评测 | 47 小时 35 分 00 秒 |
| 成功主流程与全量评测时间合计 | 57 小时 12 分 36 秒 |

## 结论

工程链路已经跑通：原生 backend 能承载 108 个 TT Linear，TT-only 训练、checkpoint、自动续训、final 导出和离线重载都可以正常工作。

模型效果不理想。完整模型虽然少了 28.67% 的参数，但五项任务都明显落后于 Dense；healing 只在 BoolQ 和 HellaSwag 上补回一部分分数。当前 native backend 的评测速度也比 Dense 慢得多，参数量下降没有直接带来推理收益。

这组实验没有覆盖正式对话质量、安全性和长上下文能力，也不足以比较 D=96、block 18–35 和 TT-only healing 是否是最佳组合。现有结果已经足以淘汰“保持当前配置，只增加 Alpaca 训练量”这条路线。

## 下一轮

下一轮把时间主要花在选层、backend 和 healing 上，不再为每个中间方案跑五任务全量评测。

1. **先做 Qwen3-8B 的层敏感性分析。** 参考 CompactifAI 的做法，分别考察不同 block 和不同投影矩阵在 MPO 压缩后的能力变化，确认 Qwen3-8B 哪些层更耐压缩、哪些层应该保留 Dense。当前直接采用后半段 block 18–35，是从 LLaMA-2 7B 的结论迁移过来的；下一轮要用 Qwen3-8B 自己的结果选层。
2. **评估是否更换第三方 TT/MPO backend。** 用相同权重、modes 和 ranks 对比 native backend 与 TensorLy-Torch 等实现，分别测前向、反向、峰值显存和数值误差。当前 native 实现的参数更少，但训练和推理没有变快；是否切换 backend 由实测结果决定。
3. **增加 healing，减少中间评测。** 把 healing 从当前 600 steps、约 0.23 epoch 提高到更完整的训练量，并保留阶段 checkpoint 观察恢复趋势。配置筛选阶段只跑少量任务或小样本评测，例如 BoolQ/HellaSwag smoke；确定最终候选后再跑五任务全量评测。这样可以把 GPU 时间更多地留给能力恢复，而不是重复评测明显不成熟的模型。
