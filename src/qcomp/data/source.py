"""统一加载 Hugging Face、JSONL 和本地磁盘数据集。

本模块用三个不可变配置对象描述数据来源，并通过 ``load_dataset_source`` 返回统一的
Hugging Face Dataset 对象。这里只处理数据位置和 split 选择，不执行 tokenize、prompt
构造、batching 或指标计算；``datasets`` 保持懒加载，不影响 qcomp 核心包导入。

主要内容：
- ``HuggingFaceDatasetSource``：配置 Hub 数据集及可选子配置和缓存路径。
- ``JsonlDatasetSource``：配置本地 JSONL 文件和逻辑 split。
- ``LocalDatasetSource``：配置 ``Dataset.save_to_disk`` 产生的本地目录。
- ``DatasetSource``：当前支持的数据来源联合类型。
- ``load_dataset_source``：把任一来源配置加载为一个具体 Dataset split。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from ..runtime import configure_runtime


@dataclass(frozen=True)
class HuggingFaceDatasetSource:
    """描述一个 Hugging Face Hub 数据集 split。"""

    dataset: str
    split: str = "train"
    config: str | None = None
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """规范名称并确认数据集与 split 不为空。

        异常：
            ValueError: 数据集名称或 split 为空时抛出。
        """

        dataset = self.dataset.strip()
        split = self.split.strip()
        if not dataset or not split:
            raise ValueError("dataset and split must not be empty")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)
        if self.config is not None:
            config = self.config.strip()
            object.__setattr__(self, "config", config or None)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())


@dataclass(frozen=True)
class JsonlDatasetSource:
    """描述一个作为单个 Dataset split 加载的本地 JSONL 文件。"""

    path: str | Path
    split: str = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """规范路径并确认 JSONL 文件和 split 有效。

        异常：
            FileNotFoundError: JSONL 文件不存在时抛出。
            ValueError: split 为空时抛出。
        """

        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file does not exist: {path}")
        split = self.split.strip()
        if not split:
            raise ValueError("split must not be empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "split", split)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())


@dataclass(frozen=True)
class LocalDatasetSource:
    """描述一个由 Hugging Face Datasets 保存到磁盘的数据集目录。"""

    path: str | Path
    split: str | None = None

    def __post_init__(self) -> None:
        """规范路径并确认本地数据集目录存在。

        异常：
            FileNotFoundError: 本地数据集目录不存在时抛出。
        """

        path = Path(self.path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"local dataset directory does not exist: {path}")
        object.__setattr__(self, "path", path)
        if self.split is not None:
            split = self.split.strip()
            object.__setattr__(self, "split", split or None)


DatasetSource: TypeAlias = (
    HuggingFaceDatasetSource | JsonlDatasetSource | LocalDatasetSource
)


def load_dataset_source(
    source: DatasetSource,
    *,
    runtime_config_path: str | Path | None = None,
) -> Any:
    """根据统一来源配置加载一个 Hugging Face Dataset split。

    参数：
        source: Hub、JSONL 或本地磁盘数据集配置。
        runtime_config_path: 可选 runtime TOML；省略时读取项目默认配置。

    返回：
        可由调用方继续预处理的 Hugging Face Dataset。

    异常：
        KeyError: 本地 DatasetDict 不包含请求的 split 时抛出。
        ValueError: 本地 DatasetDict 没有指定 split 时抛出。
        ModuleNotFoundError: 当前环境没有安装 ``datasets`` 时抛出。
    """

    runtime = configure_runtime(runtime_config_path)

    from datasets import DatasetDict, DownloadConfig, load_dataset, load_from_disk

    if isinstance(source, HuggingFaceDatasetSource):
        return load_dataset(
            source.dataset,
            source.config,
            split=source.split,
            cache_dir=str(source.cache_dir or runtime.datasets_cache),
            download_config=DownloadConfig(local_files_only=runtime.offline),
        )
    if isinstance(source, JsonlDatasetSource):
        return load_dataset(
            "json",
            data_files={source.split: str(source.path)},
            split=source.split,
            cache_dir=str(source.cache_dir or runtime.datasets_cache),
        )

    dataset = load_from_disk(str(source.path))
    if isinstance(dataset, DatasetDict):
        if source.split is None:
            raise ValueError("split is required when loading a local DatasetDict")
        if source.split not in dataset:
            raise KeyError(f"local DatasetDict does not contain split {source.split!r}")
        return dataset[source.split]
    return dataset
