"""把本地文本文档打包为 Causal LM 训练 DataLoader。

本模块从 data source 加载记录，读取调用方指定的文本字段，对每篇文档执行 tokenize 并
追加 EOS，再把连续 token 流切成训练 block。这里只准备 ``input_ids``、
``attention_mask`` 和 ``labels``；benchmark prompt、答案处理和指标计算属于 lm-eval。

主要内容：
- ``DataLoaderConfig``：配置训练 batch、block 长度、部分选择和随机顺序。
- ``build_causal_lm_dataloader``：构造可直接交给训练循环的 PyTorch DataLoader。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .source import DatasetSource, load_dataset_source


@dataclass(frozen=True)
class DataLoaderConfig:
    """定义 Causal LM 训练 DataLoader 的 packing 和 batching 参数。"""

    batch_size: int = 1
    max_length: int = 2048
    shuffle: bool = False
    max_blocks: int | None = None
    seed: int = 42
    drop_remainder: bool = False
    pin_memory: bool = False

    def __post_init__(self) -> None:
        """校验长度、部分选择和布尔配置。

        异常：
            TypeError: 布尔开关类型无效时抛出。
            ValueError: batch、block 长度或 block 数量无效时抛出。
        """

        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        if self.max_blocks is not None and self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        for name in ("shuffle", "drop_remainder", "pin_memory"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


class _TokenBlockDataset(Dataset[Mapping[str, list[int]]]):
    """保存已经完成 packing 的 token blocks。"""

    def __init__(self, blocks: Sequence[Sequence[int]]) -> None:
        """复制并校验非空 token blocks。

        参数：
            blocks: 按连续文档 token 流切分的非空 block。

        异常：
            ValueError: 没有 block 或存在空 block 时抛出。
        """

        self.blocks = tuple(tuple(int(token) for token in block) for block in blocks)
        if not self.blocks or any(not block for block in self.blocks):
            raise ValueError("token blocks must be non-empty")

    def __len__(self) -> int:
        """返回 token block 数量。"""

        return len(self.blocks)

    def __getitem__(self, index: int) -> Mapping[str, list[int]]:
        """返回一个 token block 及其 attention mask。

        参数：
            index: block 下标。

        返回：
            包含 input IDs 和全一 attention mask 的映射。
        """

        input_ids = list(self.blocks[index])
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


class _EpochRandomSampler(Sampler[int]):
    """按 seed 和 epoch 生成可恢复的确定性随机顺序。"""

    def __init__(self, data_source: Dataset[Any], seed: int) -> None:
        """保存数据集和基础随机种子。

        参数：
            data_source: 待采样的数据集。
            seed: epoch 零使用的随机种子。
        """

        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，使断点续训可以重建相同顺序。

        参数：
            epoch: 非负 epoch 下标。

        异常：
            ValueError: epoch 为负数时抛出。
        """

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        """按照当前 epoch 的确定性排列迭代 block 下标。"""

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        """返回每个 epoch 的采样数量。"""

        return len(self.data_source)


class _CausalLMCollator:
    """使用 tokenizer padding，并从 input IDs 构造训练 labels。"""

    def __init__(self, tokenizer: Any) -> None:
        """保存具有 padding token 的 tokenizer。

        参数：
            tokenizer: 提供 ``pad`` 和 ``pad_token_id`` 的 tokenizer。

        异常：
            ValueError: tokenizer 没有 padding token 时抛出。
        """

        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer must define pad_token_id")
        self.tokenizer = tokenizer

    def __call__(
        self,
        features: Sequence[Mapping[str, Sequence[int]]],
    ) -> Mapping[str, Tensor]:
        """把 token blocks padding 为 batch 并屏蔽 padding labels。

        参数：
            features: 非空 token block 序列。

        返回：
            包含 input_ids、attention_mask 和 labels 的 batch。

        异常：
            ValueError: features 为空时抛出。
        """

        if not features:
            raise ValueError("features must not be empty")
        batch = dict(
            self.tokenizer.pad(features, padding=True, return_tensors="pt")
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def _build_token_blocks(
    dataset: Any,
    tokenizer: Any,
    text_field: str,
    config: DataLoaderConfig,
) -> list[list[int]]:
    """把数据集中的文档连续打包为 token blocks。

    达到 ``max_blocks`` 后立即停止遍历和 tokenize 后续记录；最后不足一个 block 的
    token 是否保留由 ``drop_remainder`` 决定。

    参数：
        dataset: 迭代时产生字段映射的数据集。
        tokenizer: 提供 ``encode`` 和 ``eos_token_id`` 的 tokenizer。
        text_field: 每条记录中保存正文的字段名。
        config: block 长度、数量上限和尾部处理配置。

    返回：
        按文档连续 token 流切分的 blocks。

    异常：
        ValueError: tokenizer 没有 EOS、记录字段无效或没有产生可用 block 时抛出。
    """

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    blocks: list[list[int]] = []
    pending: list[int] = []
    for record in dataset:
        if not isinstance(record, Mapping):
            raise ValueError("dataset records must be mappings")
        text = record.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"record field {text_field!r} must be a string")
        pending.extend(
            int(token)
            for token in tokenizer.encode(text, add_special_tokens=False)
        )
        pending.append(int(eos_token_id))

        while len(pending) >= config.max_length:
            blocks.append(pending[: config.max_length])
            del pending[: config.max_length]
            if config.max_blocks is not None and len(blocks) >= config.max_blocks:
                return blocks

    if pending and not config.drop_remainder:
        blocks.append(pending)
    if not blocks:
        raise ValueError("dataset produced no token blocks")
    return blocks


def build_causal_lm_dataloader(
    source: DatasetSource,
    tokenizer: Any,
    config: DataLoaderConfig,
    *,
    text_field: str = "text",
    runtime_config_path: str | Path | None = None,
) -> DataLoader[Mapping[str, Tensor]]:
    """从本地数据来源构造 Causal LM 训练 DataLoader。

    参数：
        source: 本地文件、本地 Dataset 或严格离线缓存的数据来源。
        tokenizer: 提供 encode、pad 和 EOS/padding token IDs 的 tokenizer。
        config: batch、block packing、部分选择和 shuffle 配置。
        text_field: 原始记录中保存完整训练文本的字段名。
        runtime_config_path: 可选 runtime TOML；省略时读取项目默认配置。

    返回：
        产生 input_ids、attention_mask 和 labels 的训练 DataLoader。

    异常：
        ValueError: 文本字段名称为空或数据无法构造训练 block 时抛出。
    """

    normalized_text_field = text_field.strip()
    if not normalized_text_field:
        raise ValueError("text_field must not be empty")
    dataset = load_dataset_source(
        source,
        runtime_config_path=runtime_config_path,
    )
    blocks = _build_token_blocks(
        dataset,
        tokenizer,
        normalized_text_field,
        config,
    )
    tokenized = _TokenBlockDataset(blocks)
    sampler = (
        _EpochRandomSampler(tokenized, config.seed) if config.shuffle else None
    )
    return DataLoader(
        tokenized,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=_CausalLMCollator(tokenizer),
        pin_memory=config.pin_memory,
    )
