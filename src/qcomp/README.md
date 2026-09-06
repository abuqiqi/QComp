# `qcomp` 源码结构

本目录是 `qcomp` Python 包的源码入口。`runtime.py`、`model.py`、`logging.py` 和 `storage.py`
提供跨模块使用的通用功能；各子目录分别负责张量网络表示、PyTorch 模型层、计算后端、训练、workflow 和评测。

当前代码支持单个或多个无 bias Linear 的 MPO 分解、混合表示压缩计划、原子模型层
替换与恢复，并能使用外部 backend 和 evaluator 执行多指标敏感性分析。项目也支持
通过 `run_sensitivity_experiment()` 自动组装逐层 lm-eval 实验，执行
Causal LM 正常生成，以及对已经安装一个或多个张量网络层的模型进行微调和断点续训。

## 目录结构

```text
qcomp/
├── README.md          # 源码结构和职责说明
├── __init__.py        # 导出常用公共接口
├── logging.py         # 统一 JSON Lines 实验日志
├── runtime.py         # 默认 runtime.toml、缓存路径和严格离线环境
├── model.py           # 加载 Causal LM，并操作模型中的 Linear
├── storage.py         # ArtifactPaths 及 TensorNetworkArtifact 读写
├── data/              # 训练数据来源和 Causal LM DataLoader
├── representations/   # 张量网络数据格式和数学操作
├── nn/                # 张量网络 PyTorch 模型层的基类和公共行为
├── backends/          # 不同计算库的分解和具体模型层实现
├── evaluation/        # 压缩、指标方向、性能和通用 lm-eval 评测
├── training/          # 通用训练循环、objective 和 checkpoint
└── workflows/         # 压缩、敏感性分析、推理和微调流程编排
```

## 顶层文件

### `logging.py`

`log_event(path, event, **fields)` 自动创建父目录，将 UTC 时间、事件名称和
`fields` 对象序列化为一行 UTF-8 JSON，追加写入后关闭文件。字段使用 JSON 支持的
类型及有限数值；同一文件保留历次记录，每轮实验可选择独立文件名。

```python
from qcomp import SensitivityCaseResult, log_event, sensitivity_case_record

path = "artifacts/evaluations/experiment.jsonl"
log_event(path, "experiment_started", task="mmlu")

def on_case_result(result: SensitivityCaseResult) -> None:
    """将敏感性结果参数 result 整理后写入实验日志。"""
    log_event(path, "case_completed", **sensitivity_case_record(result))


log_event(
    "artifacts/evaluations/backend_comparison.jsonl",
    "backend_benchmark_completed",
    backend="native",
    latency_ms=1.23,
)
```

将 `on_case_result` 传给 `analyze_sensitivity()`，即可在每个 case 恢复模型后记录
质量指标、退化量、逐层压缩率与误差、模型压缩率及耗时。

### `runtime.py`

`runtime.py` 读取项目根目录的 `config/runtime.toml`，统一提供模型来源、Hugging Face
home、datasets cache 和离线模式。`load_runtime_config()` 只解析配置；模型、数据或
lm-eval 开始 I/O 时调用 `configure_runtime()` 应用当前进程环境。

### `model.py`

`model.py` 使用 Transformers 通用接口加载单设备 Causal LM 和 tokenizer，并
操作已经加载的 PyTorch 模型结构。省略模型来源时读取项目 `config/runtime.toml`；
Transformers 在调用加载函数时才会导入。

主要公共对象：

- `ModelLoadConfig`：定义模型路径或 ID、设备、dtype 和远程代码选项。
- `CausalLMResources`：组合加载后的模型与 tokenizer。
- `load_causal_lm()`：加载模型，并将模型移动到指定设备。
- `list_linears()`：列出模型中具有模块路径的无 bias `torch.nn.Linear`。
- `find_linear()`：按模块路径取得一个无 bias Linear。
- `replace_linear()`：将目标 Linear 替换为 backend 已经构造好的 `TensorNetworkLinear`。
- `restore_linear()`：关闭压缩模型层并恢复原始 Linear。
- `LinearReplacement`：保存一次替换所需的目标路径、原始层和压缩层。

```python
import torch

from qcomp import ModelLoadConfig, load_causal_lm

resources = load_causal_lm(
    ModelLoadConfig(
        device="cuda:0",
        dtype=torch.bfloat16,
    )
)
model = resources.model
tokenizer = resources.tokenizer
```

### `storage.py`

`storage.py` 负责持久化 representations 层定义的通用
`TensorNetworkArtifact`，并根据显式根目录提供标准运行产物路径。

主要公共对象：

- `ArtifactPaths`：提供 datasets、cache、decompositions、checkpoints 和 evaluations 路径。
- `save_artifact()`：保存表示名称、格式版本、元数据和命名张量。
- `load_artifact()`：加载上述数据，并恢复为 `TensorNetworkArtifact`。

### `__init__.py`

`__init__.py` 汇总常用公共接口，使调用者可以直接从 `qcomp` 导入。可选 backend 依赖仍保持懒加载，不会因为导入 `qcomp` 而全部加载。

## 相关目录

- [`data/`](data/README.md)：加载本地数据并自动构造通用训练 DataLoader。
- [`representations/`](representations/README.md)：定义 `TensorNetworkArtifact`、`MPOSpec`、canonical MPO 格式和稠密重建。
- [`nn/`](nn/README.md)：定义 `TensorNetworkLinear`、MPO Linear 基类和公共模型层行为。
- [`backends/`](backends/README.md)：实现 native、TensorLy、torchTT 和 cuTensorNet 的 MPO 适配器与具体模型层。
- [`evaluation/`](evaluation/README.md)：计算压缩指标、常用指标方向、通用 lm-eval benchmark 和性能时间。
- [`training/`](training/README.md)：定义通用 Causal LM 训练循环、loss objective 和 checkpoint。
- [`workflows/`](workflows/README.md)：提供单层与模型级压缩、恢复、多指标敏感性分析、正常生成和张量网络微调流程。
