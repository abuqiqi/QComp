"""Tokenization, blocking, collation, and deterministic sampling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from ..provenance import canonical_json_sha256
from .base import DocumentSource, PreparedCausalLMData


class TokenBlockDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, blocks: Sequence[Sequence[int]]) -> None:
        self.blocks = [list(map(int, block)) for block in blocks]
        if not self.blocks or any(not block for block in self.blocks):
            raise ValueError("token blocks must be non-empty")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        ids = self.blocks[index]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class CausalLMCollator:
    def __init__(self, tokenizer: Any) -> None:
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer must define pad_token_id")
        self.tokenizer = tokenizer

    def __call__(
        self, features: Sequence[Mapping[str, Sequence[int]]]
    ) -> dict[str, Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        batch = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class EpochRandomSampler(Sampler[int]):
    def __init__(self, data_source: Dataset[Any], *, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(
            torch.randperm(
                len(self.data_source),
                generator=torch.Generator().manual_seed(self.seed + self.epoch),
            ).tolist()
        )

    def __len__(self) -> int:
        return len(self.data_source)


def prepare_causal_lm_data(
    source: DocumentSource,
    tokenizer: Any,
    *,
    max_length: int = 1024,
    batch_size: int = 1,
    seed: int = 42,
    max_blocks: int | None = None,
    drop_remainder: bool = False,
    pin_memory: bool = False,
) -> PreparedCausalLMData:
    if max_length <= 0 or batch_size <= 0:
        raise ValueError("max_length and batch_size must be positive")
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    stream: list[int] = []
    documents = 0
    for text in source.documents():
        if not isinstance(text, str):
            raise TypeError("document sources must yield strings")
        stream.extend(map(int, tokenizer(text, add_special_tokens=False)["input_ids"]))
        stream.append(int(eos))
        documents += 1
    if not stream:
        raise ValueError("document source produced no tokens")
    blocks = [
        stream[start : start + max_length]
        for start in range(0, len(stream), max_length)
    ]
    if drop_remainder and blocks and len(blocks[-1]) != max_length:
        blocks.pop()
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    dataset = TokenBlockDataset(blocks)
    sampler = EpochRandomSampler(dataset, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=CausalLMCollator(tokenizer),
        pin_memory=pin_memory,
    )
    descriptor = {
        "source_fingerprint": source.fingerprint(),
        "source": dict(source.metadata()),
        "documents": documents,
        "full_tokens": len(stream),
        "selected_tokens": sum(map(len, blocks)),
        "blocks": len(blocks),
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": seed,
        "max_blocks": max_blocks,
        "drop_remainder": drop_remainder,
    }
    return PreparedCausalLMData(
        dataloader, canonical_json_sha256(descriptor), descriptor
    )
