"""公开数据来源和通用 Causal LM 训练 DataLoader 接口。

本包统一加载 Hugging Face cache、本地 JSONL 和磁盘 Dataset，并把指定文本字段连续
打包为 Causal LM 训练 batch。本包不定义 benchmark prompt 或评测指标；标准评测由
lm-eval 负责。可选的 ``datasets`` 依赖只在实际加载时导入。

主要内容：
- ``DatasetSource``：组合当前支持的三种数据来源配置类型。
- ``HuggingFaceDatasetSource``：描述 Hub 数据集、配置、split 和缓存目录。
- ``JsonlDatasetSource``：描述本地 JSONL 文件及其逻辑 split。
- ``LocalDatasetSource``：描述 ``load_from_disk`` 保存的本地数据集。
- ``load_dataset_source``：根据配置加载并返回 Hugging Face Dataset。
- ``DataLoaderConfig``、``build_causal_lm_dataloader``：构造通用训练 batch。
"""

from .preprocessing import (
    DataLoaderConfig,
    build_causal_lm_dataloader,
)
from .source import (
    DatasetSource,
    HuggingFaceDatasetSource,
    JsonlDatasetSource,
    LocalDatasetSource,
    load_dataset_source,
)

__all__ = [
    "DataLoaderConfig",
    "DatasetSource",
    "HuggingFaceDatasetSource",
    "JsonlDatasetSource",
    "LocalDatasetSource",
    "build_causal_lm_dataloader",
    "load_dataset_source",
]
