# Qwen3 TT Compression

这是一个面向 causal language model 的 TT-matrix 研究仓库。它完成四件事：加载本地
Dense 模型和训练数据，把指定 Linear 层分解并替换为 TT-matrix，只训练 TT cores，
最后通过 `lm-eval` 比较替换前、训练前和训练后的模型指标。

```text
本地 Dense 模型 + 训练数据
        ↓
分解指定层，生成 TT module set
        ↓
用 TT 层替换 Dense 层，只训练 TT cores
        ↓
评测 Dense / TT-before / TT-after
```

模型目标由配置中的 `module_path` 描述，不依赖固定的 Qwen 层路径。TT checkpoint 保存
为与运行后端无关的 canonical cores，同一份 module set 可以交给不同 TT 后端加载。

## 安装

在包含 `pyproject.toml` 的仓库根目录执行：

```bash
python -m pip install -e .
```

默认安装已经包含 `native`（原生 PyTorch）后端，不需要额外的 TT 库。其他后端按需
安装各自的 optional dependency；当前仓库额外提供了 TensorLy-Torch 适配器：

```bash
python -m pip install -e ".[tensorly]"
```

也提供了 torchTT 适配器；依赖由使用者按需安装：

```bash
python -m pip install -e ".[torchtt]"
```

cuTensorNet 适配器目前只支持 inference，大输入走缓存的 cuTensorNet contraction，
小输入自动回退 native MPO：

```bash
python -m pip install -e ".[cutensornet]"
```

后端在配置文件中用名字选择，例如 `"native"`、`"tensorly_torch"`、`"torchtt"`
或 `"cutensornet"`。这些都是仓库内置适配器、外部可选依赖，checkpoint 格式不与
其中任何一个绑定。cuTensorNet 可以通过普通 `backend_options` 调整切换阈值和缓存：

```json
{
  "backend": "cutensornet",
  "backend_options": {
    "min_tokens": 16,
    "token_bucket_size": 128,
    "max_cached_networks": 2,
    "memory_limit": "256MiB"
  }
}
```

`min_tokens` 以下继续使用 native MPO，适合单 token decode；较长输入按
`token_bucket_size` 分桶并缓存 contraction plan。cuTensorNet backend 在
`trainable=True` 时会明确报错。若环境同时包含不同 CUDA 12.x minor 版本，应保证
cuTensorNet、cuBLAS 和 cuSOLVER 从同一套 Toolkit 加载。

## 新增 MPO Backend

新的 backend 可以放在本仓库外，不需要修改内置 registry、模型替换逻辑或 checkpoint
格式。它只需要实现 `TTLinearBase`、实现一个 `TTBackend` factory，并在模块导入时显式
注册。下面假设第三方包中有 `my_package/qwen3_backend.py`：

```python
from typing import Any, Mapping, Sequence

from torch import Tensor

from qwen3_tn import register_tt_backend
from qwen3_tn.backends import ParameterListTTLinear, TTBackend
from qwen3_tn.tt import TTMatrixSpec

import my_mpo_kernel


class MyMPOLinear(ParameterListTTLinear):
    def __init__(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int,
        trainable: bool,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
        kernel: str,
        autotune: bool,
    ) -> None:
        super().__init__(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
        )
        self.kernel = kernel
        self.autotune = autotune

    @property
    def backend_name(self) -> str:
        return "my_mpo"

    @property
    def backend_version(self) -> str:
        return "1"

    def _forward_impl(self, inputs: Tensor) -> Tensor:
        # 用自己的 Triton/CUDA/第三方 MPO contraction 替换这里。
        return my_mpo_kernel.linear(
            inputs,
            tuple(self.cores),
            self.spec,
            kernel=self.kernel,
            autotune=self.autotune,
            token_chunk_size=self.token_chunk_size,
        )


class MyMPOBackend(TTBackend):
    name = "my_mpo"

    @property
    def backend_version(self) -> str:
        return "1"

    def build_linear(
        self,
        spec: TTMatrixSpec,
        cores: Sequence[Tensor],
        *,
        token_chunk_size: int,
        trainable: bool,
        preserve_input_dtype: bool,
        activation_checkpointing: bool,
        backend_options: Mapping[str, Any],
    ) -> TTLinearBase:
        options = dict(backend_options)
        kernel = str(options.pop("kernel", "fused"))
        autotune = bool(options.pop("autotune", False))
        if options:
            raise ValueError(f"unknown my_mpo options: {sorted(options)}")
        if trainable and not my_mpo_kernel.supports_backward:
            raise RuntimeError("my_mpo backend does not support training")
        return MyMPOLinear(
            spec,
            cores,
            token_chunk_size=token_chunk_size,
            trainable=trainable,
            preserve_input_dtype=preserve_input_dtype,
            activation_checkpointing=activation_checkpointing,
            kernel=kernel,
            autotune=autotune,
        )


# 类本身就是零参数 factory；模块被 import 后完成注册。
register_tt_backend("my_mpo", MyMPOBackend)
```

`spec` 和 `cores` 使用仓库的 canonical 格式，第 `i` 个 core 的形状是
`[r_i, out_mode_i, in_mode_i, r_(i+1)]`。后端可以在内部生成 packed layout 或
autotune cache，但 `export_cores()` 和 `load_cores()` 必须保持 canonical 顺序和
形状，这样同一份普通 TT module set 才能跨 backend 使用。backend 与 layer 返回的
`name/version` 必须一致；改变 kernel 数值语义或持久化格式时应升级 version。

先确保第三方包已经安装或位于 `PYTHONPATH`，然后在普通训练或评测配置中写：

```json
{
  "backend": "my_mpo",
  "backend_modules": ["my_package.qwen3_backend"],
  "backend_options": {
    "kernel": "fused",
    "autotune": true
  }
}
```

CLI 会先 import `backend_modules`，再解析 `backend`。直接使用 Python API 时，可以
先 import 注册模块，然后正常调用 `install_tt_modules(..., tt_backend="my_mpo",
backend_options={...})`。`registered_tt_backends()` 检查名字是否完成注册；
`available_tt_backends()` 只返回依赖和运行条件均满足的后端，单个后端的不可用原因可通过
`probe_tt_backend(name)` 查看。
`replace=True` 仅建议用于交互式开发时重新注册同名 backend。

多 backend benchmark 的 options 要按 backend 名称分组：

```json
{
  "backends": ["native", "my_mpo"],
  "backend_modules": ["my_package.qwen3_backend"],
  "backend_options": {
    "my_mpo": {
      "kernel": "fused",
      "autotune": true
    }
  }
}
```

```bash
qwen3-tn-benchmark inference configs/experiments/my-mpo-benchmark.json
```

接入时至少检查以下几点：

1. `trainable=False` 的 inference 实例不保存 Dense weight，也不在 forward 中重建
   Dense weight。
2. 支持训练时，外部 kernel 必须有 autograd/backward；不支持时应在
   `trainable=True` 时给出清晰错误。
3. 未知 `backend_options` 应明确报错，避免拼写错误被静默忽略。
4. 先通过单层 forward/backward 与 checkpoint round-trip，再运行全模型
   `1024 prefill + 32 decode` benchmark。只有数值门槛通过且端到端达到目标加速后，
   才继续运行小样本和完整 `lm-eval`。

## 从 Dense 模型到训练后的 TT 模型

常规实验只需要两个命令：先用 `qwen3-tn-decompose` 生成 TT module set，再用
`qwen3-tn-finetune` 加载模型和数据、替换目标层、训练 TT cores 并评测。命令只接收
一个 JSON 配置路径；仓库中的 `configs/` 提供了可复制修改的完整示例。

### 1. 准备模型、训练数据和评测任务

模型必须已经下载到本机。配置中的 `model_path` 指向模型目录，命令会通过
Transformers 加载模型和 tokenizer，并使用 `local_files_only=True`，不会自动下载
模型。

训练数据支持两种来源。可以使用 Hugging Face Dataset：

```json
{
  "type": "huggingface",
  "dataset": "EleutherAI/wikitext_document_level",
  "config": "wikitext-2-raw-v1",
  "split": "train",
  "text_field": "page"
}
```

也可以使用本地 JSONL；每行是一个 JSON 对象，默认从 `text` 字段读取文本：

```json
{
  "type": "jsonl",
  "path": "/path/to/train.jsonl",
  "text_field": "text"
}
```

训练数据与评测数据是两回事。评测由 `lm-eval` 按任务名加载，例如
`configs/evaluation/hellaswag.json`：

```json
{
  "task": "hellaswag",
  "metrics": [{"name": "acc_norm,none", "direction": "higher"}],
  "limit": null,
  "batch_size": 1,
  "max_length": 2048,
  "num_fewshot": 0,
  "apply_chat_template": false,
  "bootstrap_iters": 0,
  "seed": 42
}
```

Hugging Face 训练数据和 `lm-eval` 评测任务如果本地没有缓存，可能需要联网下载；本地
JSONL 不需要。

### 2. 分解需要替换的层

复制并修改 `configs/compression/qwen3_8b_down_proj_r2048.json`。其中最重要的字段是：

```json
{
  "model_path": "/path/to/local/model",
  "output_root": "artifacts/decompositions/my-experiment",
  "cache_root": "artifacts/cache/decompositions",
  "device": "cuda:0",
  "targets": [
    {
      "module_path": "model.layers.0.mlp.down_proj",
      "out_modes": [8, 8, 8, 8],
      "in_modes": [8, 8, 8, 24],
      "ranks": [1, 64, 256, 192, 1],
      "token_chunk_size": 8
    }
  ]
}
```

`module_path` 是要替换的 bias-free `torch.nn.Linear`。`out_modes` 和 `in_modes` 的乘积
必须分别等于该层的 `out_features` 和 `in_features`；`ranks` 的长度必须比 modes 多
一个，并且首尾都是 `1`。

```bash
qwen3-tn-decompose configs/compression/my-experiment.json
```

命令加载 Dense 模型，对每个 target 执行 TT-SVD，并在 `output_root` 下生成：

```text
artifacts/decompositions/my-experiment/
├── index.json
└── modules/                       # 各目标层的 canonical TT cores
```

这个目录称为 TT module set。`index.json` 记录模型层路径及其对应的 cores，后续替换和
训练都以它为输入。

### 3. 替换模型层并训练 TT cores

复制并修改 `configs/experiments/qwen3_8b_wikitext_finetune.json`，把上一步的 module
set、训练数据和评测任务组合到一次实验中：

```json
{
  "model_path": "/path/to/local/model",
  "module_set": "artifacts/decompositions/my-experiment/index.json",
  "artifact_root": "artifacts/finetuned/my-experiment",
  "result_root": "results/finetune/my-experiment",
  "backend": "native",
  "data": {
    "type": "huggingface",
    "dataset": "EleutherAI/wikitext_document_level",
    "config": "wikitext-2-raw-v1",
    "split": "train",
    "text_field": "page"
  },
  "training": {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "max_length": 1024,
    "device": "cuda:0"
  },
  "evaluation": "configs/evaluation/hellaswag.json"
}
```

```bash
qwen3-tn-finetune configs/experiments/my-experiment.json
```

这个命令会依次完成：

1. 加载本地 Dense 模型、tokenizer 和训练数据。
2. 如果配置了 `evaluation`，先评测原始 Dense 模型。
3. 按 module set 的 `module_path` 将 Dense 层替换为 TT 层，并评测训练前的 TT 模型。
4. 冻结 backbone，只让 FP32 TT cores 参与优化。
5. 训练结束后导出 BF16 canonical cores，并评测训练后的 TT 模型。

CUDA 前向默认使用 BF16 autocast。TT contraction activation checkpointing 默认开启；
模型级 gradient checkpointing、保存频率等行为由 `training` 配置控制。

### 4. 查看训练和评测结果

主要产物位于：

```text
artifacts/finetuned/my-experiment/
├── checkpoint-*/                  # optimizer、scheduler、训练位置和 RNG，可用于恢复
├── final/index.json               # 训练后的 TT module set
└── run.json                       # 本次训练的签名和指标

results/finetune/my-experiment/
├── baseline.json                  # 原始 Dense 模型
├── before.json                    # 未微调的 TT 模型
├── after.json                     # 微调后的 TT 模型
└── comparison.json                # 三者的指标对比
```

未配置 `evaluation` 时，只训练和导出 TT cores，不生成上述四个评测文件。

配置里的相对路径按执行命令时的当前目录解析；直接使用仓库自带配置时，建议在仓库
根目录运行命令。

## 自定义和进阶功能

常规实验不需要直接调用 Python API。只有在 CLI 不能满足需求时，例如需要自定义
`DocumentSource`、模型加载过程或 evaluator，才需要组合以下模块：

- `qwen3_tn.workflows`：完整的分解、微调、评测和 rank sweep 流程。
- `qwen3_tn.model`：安装、临时替换和导出 TT 模块。
- `qwen3_tn.data`：Hugging Face、JSONL 和自定义训练数据源。
- `qwen3_tn.training`：TT-only 训练器。
- `qwen3_tn.evaluation`：基于 `lm-eval` 的单模型评测。
- `qwen3_tn.tt` / `qwen3_tn.backends`：TT 数学和运行后端，主要面向后端开发。

其余命令也属于可选工具：

| 命令 | 什么时候使用 |
| --- | --- |
| `qwen3-tn-evaluate CONFIG` | 不训练，只单独评测 Dense 模型或已有 TT module set |
| `qwen3-tn-sweep CONFIG` | 批量分解和评测多组 rank 候选 |
| `qwen3-tn-benchmark {layer,training,inference} CONFIG` | 比较多个后端的速度、显存和数值误差 |

对应的完整配置示例位于 `configs/`；命令语法可通过 `--help` 查看，配置字段由
`src/qwen3_tn/cli/` 中的对应入口定义。

## 项目结构（开发者参考）

```text
qwen3-tn-compression/
├── pyproject.toml                 # 依赖、可选后端 extras 和 CLI entry points
├── README.md                      # 入门流程、进阶功能与仓库导航
├── configs/                       # 模型路径、TT 规格和实验参数；业务硬编码只放这里
│   ├── compression/               # 待分解 module_path、modes、ranks 和 core dtype
│   ├── data/                      # Hugging Face/JSONL 数据源与 token block 参数
│   ├── evaluation/                # lm-eval 任务、指标方向和评测参数
│   ├── experiments/               # 微调与 backend benchmark 的组合配置
│   └── sweeps/                    # 通用 rank candidate 与选择规则
├── docs/
│   ├── architecture.md            # 分层依赖、职责边界和正向数据流
│   ├── tt_matrix.md               # TT-matrix 数学、core 布局和 rank 约定
│   └── checkpoint_format.md       # 单模块、module set 与训练 checkpoint 格式
├── src/qwen3_tn/
│   ├── __init__.py                # 仅导出稳定核心 API
│   ├── provenance.py              # SHA-256、模型签名和原子 JSON 写入
│   ├── checkpoint.py              # backend-neutral canonical core 与显式 index
│   ├── model.py                   # TTTarget、安装/导出 cores 和 Dense 层回滚
│   ├── evaluation.py              # 单模型 causal-LM 评测与指标提取
│   ├── benchmarking.py            # layer/training/inference backend 基准
│   ├── tt/                        # 不依赖模型和 backend 的数学核心
│   │   ├── spec.py                # TTMatrixSpec 与 shape/rank 合法性
│   │   ├── decomposition.py       # TT-SVD
│   │   └── operations.py          # 张量化、重建、core 校验和 bond slicing
│   ├── backends/                  # 原生后端与可选后端适配器
│   │   ├── base.py                # TTLinearBase / TTBackend 协议
│   │   ├── registry.py            # 延迟注册与 create_tt_linear()
│   │   ├── native.py              # 原生 PyTorch contraction/checkpointing
│   │   └── tensorly_torch.py      # 可选 BlockTT + FactorizedLinear wrapper
│   ├── data/                      # 与具体数据集解耦的数据准备
│   │   ├── base.py                # DocumentSource 与 PreparedCausalLMData
│   │   ├── jsonl.py               # 本地 JSONL 文本源
│   │   ├── huggingface.py         # 通用 Hugging Face Dataset 文本源
│   │   └── tokenization.py        # EOS、blocking、padding、labels 和 sampler
│   ├── training/                  # 单 GPU、TT-only causal-LM 训练
│   │   ├── config.py              # 训练配置、状态和参数量摘要
│   │   ├── checkpoint.py          # optimizer/scheduler/RNG 保存恢复
│   │   └── trainer.py             # 冻结、optimizer 构建和训练循环
│   ├── workflows/                 # 可组合的高层正向研究流程
│   │   ├── decompose.py           # 内容寻址分解与 module-set 输出
│   │   ├── evaluate.py            # Dense/TT 评测与严格缓存
│   │   ├── finetune.py            # baseline → TT-before → train → TT-after
│   │   └── sweep.py               # 模型结构无关的完整 rank tuple sweep
│   └── cli/                       # 薄 CLI：只解析配置并调用 library/workflow
│       ├── decompose.py
│       ├── finetune.py
│       ├── evaluate.py
│       ├── sweep.py
│       └── benchmark.py
├── tests/                         # tiny 非 Qwen 模型上的数学、训练和流程测试
├── notebooks/                     # 现有探索性 Notebook；不作为稳定 API
├── artifacts/
│   ├── decompositions/            # 可安装的 TT module sets
│   ├── finetuned/                 # 微调 run metadata 与 final canonical cores
│   └── sweeps/                    # rank sweep baseline 和各候选 cores
└── results/
    ├── evaluations/               # 独立评测结果
    ├── finetune/                  # 微调前后指标与 comparison
    └── benchmarks/                # backend layer/training/inference 性能结果
```

核心依赖方向是：

```text
tt → backends → checkpoint + model
   → data + training + evaluation
   → workflows + benchmarking → cli
```

下层模块不依赖 workflow、CLI、具体数据集或 Qwen 层路径。`configs/` 描述一次
实验“是什么”，`workflows/` 负责组合步骤，`cli/` 只提供可执行入口；训练器本身
不会加载数据集，也不会查找或替换模型层。

详细依赖边界、数学约定和格式规范位于 `docs/`。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

未安装 TensorLy-Torch 时，native 全套测试照常运行，TensorLy 数值一致性测试跳过。
