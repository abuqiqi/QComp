# data

`data` 负责定位本地训练数据，并把文本文档转换成可直接交给 training 的 PyTorch DataLoader。它不定义 benchmark prompt、答案处理或指标；标准评测由 lm-eval 负责。

所有 Hugging Face I/O 默认读取项目根目录的 `config/runtime.toml`。当前默认配置严格离线，只读取本地 cache；`datasets` 依赖在真正加载数据时才导入。

## 数据来源

- `HuggingFaceDatasetSource`：从本地 Hugging Face cache 加载 dataset/config/split。
- `JsonlDatasetSource`：加载本地 JSONL 文件及其逻辑 split。
- `LocalDatasetSource`：加载 `save_to_disk()` 保存的 Dataset 或 DatasetDict。
- `load_dataset_source()`：只加载 Dataset，不执行 tokenize 或 batching。

```python
from qcomp import HuggingFaceDatasetSource, load_dataset_source

source = HuggingFaceDatasetSource(
    dataset="EleutherAI/wikitext_document_level",
    config="wikitext-2-raw-v1",
    split="train",
)
dataset = load_dataset_source(source)
```

JSONL 和磁盘 Dataset 使用相同入口。以下代码先创建示例文件和 Dataset 目录，来源对象构造时就会检查路径是否存在；相对路径按当前工作目录解析：

```python
from pathlib import Path
from datasets import Dataset
from qcomp import JsonlDatasetSource, LocalDatasetSource

Path("artifacts/datasets").mkdir(parents=True, exist_ok=True)
Path("artifacts/datasets/train.jsonl").write_text('{"text": "example"}\n', encoding="utf-8")
Dataset.from_dict({"text": ["example"]}).save_to_disk("artifacts/datasets/prepared")
training_source = JsonlDatasetSource("artifacts/datasets/train.jsonl")
prepared_source = LocalDatasetSource("artifacts/datasets/prepared")
```

DatasetDict 必须指定 split；单个 Dataset 会直接返回。

## 构造训练 DataLoader

`build_causal_lm_dataloader()` 读取一个明确的文本字段，对每篇文档 tokenize 并追加 EOS，把连续 token 流切成 block，然后使用 tokenizer 的标准 padding 构造 batch。`labels` 是 `input_ids` 的副本，padding 位置为 `-100`。

```python
from qcomp import (
    DataLoaderConfig,
    HuggingFaceDatasetSource,
    build_causal_lm_dataloader,
    load_causal_lm,
)

source = HuggingFaceDatasetSource(
    dataset="EleutherAI/wikitext_document_level",
    config="wikitext-2-raw-v1",
    split="train",
)
resources = load_causal_lm()
model, tokenizer = resources.model, resources.tokenizer
dataloader = build_causal_lm_dataloader(
    source,
    tokenizer,
    DataLoaderConfig(
        batch_size=4,
        max_length=2048,
        max_blocks=100,
        shuffle=True,
        seed=42,
    ),
    text_field="page",
)
```

`max_blocks` 用于部分训练数据准备；达到数量后停止读取和 tokenize 后续记录。`drop_remainder=True` 会丢弃最后一个不足 `max_length` 的 block。启用 shuffle 时，sampler 使用 `seed + epoch` 生成可复现顺序，供训练循环和断点续训调用。

接续上例，将构造结果交给训练接口。这里选取模型的第一个具名参数进行演示；实际实验应明确指定需要更新的参数：

```python
from qcomp import CausalLMObjective, TrainingConfig, train_causal_lm

result = train_causal_lm(
    model,
    dataloader,
    trainable_parameter_names=(next(iter(dict(model.named_parameters()))),),
    objective=CausalLMObjective(),
    config=TrainingConfig(max_steps=10, device="cuda:0"),
    output_dir="artifacts/checkpoints/run-1",
)
```

lm-eval 使用独立的数据处理边界：它的 task 负责 benchmark 数据、prompt、request batching、答案 filter 和 metrics，因此 `LMEvalEvaluator` 不接收训练 DataLoader。
