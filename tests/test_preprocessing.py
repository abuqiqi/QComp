"""验证通用 Causal LM 训练 DataLoader 的文本 packing 语义。

本模块使用内存记录和最小 tokenizer，检查文本字段读取、EOS 拼接、固定长度 block、
padding labels、部分预处理和按 epoch 确定的随机顺序，不加载真实数据集或模型。

主要内容：
- ``_CharacterTokenizer``：提供可预测 token ID 和 padding 的最小 tokenizer。
- ``DataLoaderBuilderTests``：验证构造出的 batch 可以直接交给训练循环。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import unittest
from unittest.mock import patch

import torch

from qcomp import DataLoaderConfig, build_causal_lm_dataloader


class _CharacterTokenizer:
    """把字符映射为字符码，并提供 EOS 和右侧 padding。"""

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """按字符生成 token IDs。

        参数：
            text: 待编码文本。
            add_special_tokens: 是否由 tokenizer 添加特殊 token。

        返回：
            每个字符对应的整数 ID。
        """

        if add_special_tokens:
            raise AssertionError("training packing must add EOS explicitly")
        return [ord(character) for character in text]

    def pad(
        self,
        features: Sequence[Mapping[str, Sequence[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]:
        """把测试 features 右侧 padding 为 PyTorch 张量。

        参数：
            features: 待组成 batch 的 token blocks。
            padding: 是否启用 padding。
            return_tensors: 返回的张量框架名称。

        返回：
            padding 后的 input IDs 和 attention mask。
        """

        if not padding or return_tensors != "pt":
            raise AssertionError("unexpected padding options")
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"])
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _CountingDataset:
    """记录 packing 实际读取了多少条原始记录。"""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        """保存测试记录并初始化读取计数。"""

        self.records = records
        self.read_records = 0

    def __iter__(self):
        """逐条返回记录并更新读取计数。"""

        for record in self.records:
            self.read_records += 1
            yield record


class DataLoaderBuilderTests(unittest.TestCase):
    """验证训练构造器不包含数据集或 benchmark 专用规则。"""

    def test_builder_packs_text_stream_and_masks_padding(self) -> None:
        """文档之间追加 EOS，连续切 block，并屏蔽 padding labels。"""

        records = [{"body": "ab"}, {"body": "cd"}]
        with patch(
            "qcomp.data.preprocessing.load_dataset_source",
            return_value=records,
        ):
            dataloader = build_causal_lm_dataloader(
                object(),
                _CharacterTokenizer(),
                DataLoaderConfig(batch_size=2, max_length=4),
                text_field="body",
            )
            batch = next(iter(dataloader))

        self.assertEqual(
            batch["input_ids"].tolist(),
            [[ord("a"), ord("b"), 2, ord("c")], [ord("d"), 2, 0, 0]],
        )
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[ord("a"), ord("b"), 2, ord("c")], [ord("d"), 2, -100, -100]],
        )

    def test_max_blocks_stops_tokenizing_remaining_records(self) -> None:
        """达到 block 上限后不再读取或 tokenize 后续文档。"""

        dataset = _CountingDataset(
            ({"text": "abcdef"}, {"text": "unused"})
        )
        with patch(
            "qcomp.data.preprocessing.load_dataset_source",
            return_value=dataset,
        ):
            dataloader = build_causal_lm_dataloader(
                object(),
                _CharacterTokenizer(),
                DataLoaderConfig(max_length=3, max_blocks=2),
            )

        self.assertEqual(len(dataloader.dataset), 2)
        self.assertEqual(dataset.read_records, 1)

    def test_shuffle_sampler_is_deterministic_per_epoch(self) -> None:
        """训练循环设置 epoch 后得到由 seed 唯一决定的 block 顺序。"""

        records = [{"text": character} for character in "abcd"]
        with patch(
            "qcomp.data.preprocessing.load_dataset_source",
            return_value=records,
        ):
            dataloader = build_causal_lm_dataloader(
                object(),
                _CharacterTokenizer(),
                DataLoaderConfig(max_length=2, shuffle=True, seed=7),
            )

        sampler = dataloader.sampler
        sampler.set_epoch(2)
        expected = torch.randperm(
            4,
            generator=torch.Generator().manual_seed(9),
        ).tolist()
        self.assertEqual(list(sampler), expected)
        sampler.set_epoch(2)
        self.assertEqual(list(sampler), expected)

    def test_invalid_config_or_training_text_is_rejected(self) -> None:
        """拒绝无效 block 配置、空字段名和非字符串训练文本。"""

        for values in (
            {"batch_size": 0},
            {"max_length": 0},
            {"max_blocks": 0},
            {"shuffle": 1},
            {"drop_remainder": 1},
            {"pin_memory": 1},
        ):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    DataLoaderConfig(**values)

        with patch(
            "qcomp.data.preprocessing.load_dataset_source",
            return_value=[{"text": 1}],
        ):
            with self.assertRaisesRegex(ValueError, "must be a string"):
                build_causal_lm_dataloader(
                    object(),
                    _CharacterTokenizer(),
                    DataLoaderConfig(),
                )
            with self.assertRaisesRegex(ValueError, "text_field"):
                build_causal_lm_dataloader(
                    object(),
                    _CharacterTokenizer(),
                    DataLoaderConfig(),
                    text_field=" ",
                )

    def test_drop_remainder_rejects_an_empty_training_set(self) -> None:
        """丢弃唯一的不完整 block 后拒绝创建空 DataLoader。"""

        with patch(
            "qcomp.data.preprocessing.load_dataset_source",
            return_value=[{"text": "a"}],
        ):
            with self.assertRaisesRegex(ValueError, "no token blocks"):
                build_causal_lm_dataloader(
                    object(),
                    _CharacterTokenizer(),
                    DataLoaderConfig(max_length=3, drop_remainder=True),
                )


if __name__ == "__main__":
    unittest.main()
