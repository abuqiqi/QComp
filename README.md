# qcomp 项目结构规划

## 1. 项目目标

`qcomp` 用于研究和实现大模型的张量网络压缩，完整流程包括：

```text
稠密模型
  → 候选方案的临时压缩与敏感性评测
  → 选出最终压缩方案
  → 正式分解与模型压缩
  → 重训练 / 微调
  → 评估与推理
```

项目需要支持不同张量网络结构，例如 MPO、Tucker、CP 和 Tensor Ring；同时允许使用 native PyTorch、TensorLy、torchTT、cuTensorNet 等不同计算库。

第一阶段只实现 MPO，但公共边界不应与 MPO 绑定，保证以后可以增加其他结构。

## 2. 设计原则

1. **张量结构与计算库分离**

   MPO、Tucker 等属于表示结构；TensorLy、torchTT 等属于具体实现后端。压缩方案描述结构，不绑定某个库。

2. **分解产物与执行模块分离**

   后端负责从稠密权重生成标准 artifact，并根据 artifact 构造可执行模块；`nn` 层定义压缩模块共有的 PyTorch 行为，artifact 本身不依赖执行方式。

3. **统一中间产物**

   不同库产生的结果必须转换成项目定义的标准 artifact，才能进行跨库验证和公平比较。

4. **工作流与底层算法分离**

   sensitivity、compress、finetune 和 inference 只编排流程，不直接实现张量分解算法。敏感性分析复用正式压缩能力，不维护另一套分解或模型替换代码。

5. **评测逻辑集中管理**

   后端只完成计算。模型质量、压缩效果、CUDA 同步、warmup、显存统计和计时范围全部由 evaluation 层统一管理。

6. **生成数据不放在 `src/` 中**

   `src/` 只保存 Python 源码；分解结果、checkpoint、缓存和实验结果统一放在项目根目录的 `artifacts/` 下。

7. **数据读取与评测规则分离**

   data 层只负责读取和预处理数据；使用哪个 split、prompt 和 metrics 由 EvaluationTask 配置决定。同一个数据集可以用于不同评测任务。

## 3. 规划目录

```text
qcomp/
├── README.md
├── AGENTS.md                       # Agent 必须遵守的代码与文档规范
├── pyproject.toml
├── configs/                         # 可复现的实验配置
│   └── evaluation/                  # 数据集、预处理和 metrics 的组合
│
├── src/qcomp/
│   ├── README.md                    # Python 包源码结构和职责说明
│   ├── representations/             # 张量网络结构及标准数据格式
│   │   ├── README.md                # 数据格式、数学操作和扩展方式
│   │   ├── artifact.py              # 与具体张量网络无关的统一 artifact
│   │   ├── registry.py              # 按表示类型分派稠密重建
│   │   └── mpo.py                   # MPO spec、core 布局和重建
│   │
│   ├── nn/                          # 可放入 PyTorch 模型的压缩模块
│   │   ├── README.md                # 公共模块接口、MPO 共享行为和扩展方式
│   │   ├── base.py                  # 通用 TensorNetworkLinear 接口
│   │   └── mpo.py                   # MPO Linear 的公共行为
│   │
│   ├── backends/                    # Provider 与表示组合的适配层
│   │   ├── base.py                  # 通用 TensorNetworkBackend 接口
│   │   ├── registry.py
│   │   ├── native/
│   │   │   └── mpo.py
│   │   ├── tensorly/
│   │   │   └── mpo.py
│   │   ├── torchtt/
│   │   │   └── mpo.py
│   │   └── cutensornet/
│   │       └── mpo.py
│   │
│   ├── model.py                     # 列出、查找、替换和恢复模型层
│   ├── storage.py                   # 通用 TensorNetworkArtifact 读写
│   │
│   ├── data/                        # 训练和评测共用的数据访问层
│   │   ├── source.py                # Hugging Face、JSONL 和本地文件
│   │   └── preprocessing.py         # tokenize、截断和 prompt 构造
│   │
│   ├── workflows/                   # 面向用户的完整流程
│   │   ├── sensitivity.py
│   │   ├── compress.py
│   │   ├── finetune.py
│   │   └── inference.py
│   │
│   ├── evaluation/                  # 模型质量与系统性能的统一评测
│   │   ├── README.md                # 当前能力、测量边界和使用示例
│   │   ├── compression.py
│   │   ├── performance.py
│   │   ├── metrics.py               # 后续按任务增加
│   │   ├── quality.py               # 后续按任务增加
│   │   └── report.py                # 后续按任务增加
│   │
│   └── cli.py                       # 单一薄命令行入口
│
├── tests/                           # 初期保持扁平，测试变多后再分类
└── artifacts/                       # 运行时生成，不提交大型文件
    ├── datasets/                    # 下载或预处理后的数据缓存
    ├── cache/
    ├── decompositions/
    ├── checkpoints/
    └── evaluations/                  # 质量、压缩与性能评测结果
```

目录和文件只在出现真实代码时创建。如果单个模块明显变大，再把它拆成同名子包，避免为尚未实现的功能预建空目录。

`AGENTS.md` 是项目的代码与文档编写规范。Agent 开始任务前必须先读取它，完成修改后再按其检查清单复核。所有 Python 模块、类、函数和方法使用中文 docstring，代码注释也统一使用中文。模块 docstring 先用一段话说明文件职责与边界，再以 `主要内容：` 和项目符号列出主要对象；对象名称使用双反引号标记，后接中文说明。

## 4. 各层职责

### representations

定义一种张量网络“是什么”，不关心由哪个库计算。

`artifact.py` 定义通用的 `TensorNetworkArtifact`，只保存 representation 名称、format version、metadata 和命名 tensors。它不假设所有张量网络都拥有相同的 ranks、cores 或 factor 结构。`registry.py` 根据 representation 调用对应的稠密重建函数，重建规则不依赖计算 backend。`storage.py` 只读写这种通用封装。

例如 `mpo.py` 负责定义：

- 输入和输出 modes；
- ranks；
- canonical core layout；
- core 校验；
- 参数量和压缩率计算；
- 重建稠密权重的参考实现。

不同表示共享 `TensorNetworkArtifact` 封装，但各自定义 metadata 和命名 tensors 的内容，不强迫 MPO、Tucker 和 CP 使用同一种 core 结构。以后增加新结构时，先增加一个文件；内容确实变多后再改成子目录。

### nn

定义压缩表示如何成为可放入 PyTorch 模型的 `nn.Module`。`base.py` 规定 artifact 导出和资源释放接口；`mpo.py` 统一 MPO Linear 的输入输出形状、core 参数访问和 artifact 导出。这里不选择 TensorLy、torchTT 等计算库。

### backends

定义一个 Provider 如何处理一种张量网络表示。`base.py` 只定义泛型 `TensorNetworkBackend[SpecT]` 和 capabilities；每个具体适配器位于 `backends/<provider>/<representation>.py`。`registry.py` 根据 Provider 与 representation 的组合进行注册、查询和懒加载。

```text
decomposition：稠密权重 → 标准 artifact
runtime：标准 artifact → 可训练或可推理的 PyTorch module
```

`decompose()` 的 spec 类型由具体表示决定。当前四个 MPO 适配器都实现 `TensorNetworkBackend[MPOSpec]`。查询时必须同时指定两个维度：`get_backend("tensorly", "mpo")`；`list_backends("mpo")` 返回支持 MPO 的 Provider。

能力通过 `BackendCapabilities` 声明，例如：

```text
native:      decomposition=yes, inference=yes, training=yes
tensorly:    decomposition=yes, inference=yes, training=yes
torchtt:     decomposition=yes, inference=yes, training=yes
cutensornet: decomposition=no,  inference=yes, training=no
```

所有 backend 共享同一种 canonical MPO artifact。TensorLy 和 torchTT 使用各自库提供的分解功能；cuTensorNet 只执行真实 CUDA contraction，不使用其他 backend 代替。

### workflows

组织完整任务，但不实现底层数学：

- `sensitivity`：为候选方案反复执行临时压缩和评测，不做重训练，最终输出报告；
- `compress`：提供统一的分解、模型替换和恢复能力，也负责根据最终 Compression Plan 生成正式压缩模型；
- `finetune`：加载压缩模型并重训练；
- `inference`：加载最终 artifact 进行推理。

`sensitivity` 必须调用 `compress` 提供的能力，退出一次候选评测后恢复原始模型。正式压缩使用同一套能力，但会保存结果并交给 `finetune`。

### data

只负责数据集的读取和通用预处理，不决定模型使用什么指标进行评测：

- `source`：统一 Hugging Face、JSONL 和本地文件等数据来源；
- `preprocessing`：tokenize、截断、样本格式转换和 prompt 构造；
- `artifacts/datasets`：保存下载或预处理缓存，真实大型数据不放入 `src/`，通常也不提交 Git。

训练和评测复用 data 层，但分别通过自己的任务配置决定 batch、split 和预处理参数。

### evaluation

统一回答“压缩后的模型到底怎么样”：

- `metrics`：定义 accuracy、perplexity、F1 等可复用指标；
- `quality`：PPL、准确率和下游任务得分；
- `compression`：参数量、张量字节数、压缩率和重建误差；
- `performance`：分解、训练、推理的时间、吞吐和显存；
- `report`：比较 dense、压缩后和重训练后三个阶段。

一次模型质量评测由 EvaluationTask 描述：

```text
EvaluationTask = Dataset + Split + Preprocessing + Metrics
```

配置放在 `configs/evaluation/`。metrics 绑定到评测任务而不是数据集本身，因此同一个数据集可以采用不同 prompt、预处理方法和指标。

性能比较仍需遵守公平原则：分解使用相同稠密权重；训练和推理使用相同标准 artifact，避免把分解误差混入运行性能比较。
压缩指标通过 representations 层的 `reconstruct_tensor()` 自动选择具体表示的重建函数，evaluation 不直接依赖 MPO、Tucker 或 CP。

### model 与 storage

`model.py` 通过 `list_linears()`、`find_linear()`、`replace_linear()` 和
`restore_linear()` 列出、查找、安装和恢复无 bias Linear。它接收 backend 已经构造
好的压缩层，不包含分解、训练或评测逻辑。`storage.py` 只负责通用
`TensorNetworkArtifact` 的保存和加载。

源码位置与实际数据位置必须区分：

```text
src/qcomp/storage.py         # Python 读写实现
artifacts/checkpoints/       # 训练产生的数据文件
```

## 5. 关键数据流

敏感性分析阶段：

```text
Dense Model
  → Candidate Compression Plan
  → 临时分解和压缩
  → Evaluation
  → 恢复 Dense Model
  → SensitivityReport
  → Final Compression Plan
```

正式压缩阶段：

```text
Dense Model
  → Final Compression Plan
  → DecompositionBackend
  → TensorNetworkArtifact
  → RuntimeBackend
  → Compressed Model
  → Fine-tuned Artifact
  → Evaluation / Inference
```

两个阶段复用相同的分解和模型替换实现，区别只是敏感性分析采用临时压缩且不重训练。`CompressionPlan` 只描述目标层、representation、modes 和 ranks；分解 Provider 与执行 Provider 由实验配置单独指定，从而支持跨库组合。

## 6. 第一阶段范围

第一阶段实现单个无 bias Linear 层的多后端闭环：

```text
Dense weight
  → Backend decomposition
  → TensorNetworkArtifact
  → Backend-built nn.Module
  → Training / Inference
  → Evaluation
  → Save / Load
```

当前支持 native、TensorLy、torchTT 和 cuTensorNet。native、TensorLy 和 torchTT 支持分解、训练与推理；cuTensorNet 支持 CUDA 推理。分解时间、推理时间和完整训练 step 时间由 evaluation 层统一测量。

当前可以按模块路径将一个无 bias `torch.nn.Linear` 替换为 backend 构造的
`TensorNetworkLinear`，执行模型 forward 后再恢复原始层。多层压缩计划和完整模型
workflow 尚未实现。

本地 Qwen3-8B 包含 253 个 `torch.nn.Linear`，这些层均不带 bias，因此第一阶段 MPO 模块统一计算 `y = xWᵀ`。

## 7. 安装与验证

使用已有 Conda 环境：

```bash
conda activate qwen3-tn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -e ".[all]"
```

最小调用示例：

```python
import torch

from qcomp import MPOSpec, get_backend
from qcomp.evaluation import compression_metrics, time_inference

weight = torch.randn(4, 4, dtype=torch.float64)
inputs = torch.randn(3, 4, dtype=torch.float64)
spec = MPOSpec.full_rank(out_modes=(2, 2), in_modes=(2, 2))

mpo_artifact = get_backend("native", "mpo").decompose(weight, spec)
linear = get_backend("tensorly", "mpo").build_linear(
    mpo_artifact,
    trainable=False,
)
outputs = linear(inputs)

metrics = compression_metrics(weight, mpo_artifact)
timing = time_inference(get_backend("native", "mpo"), mpo_artifact, inputs)
```

运行全部测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

下一步增加多层压缩计划、完整 workflow、数据集和模型质量评测。
