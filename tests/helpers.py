from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from qwen3_tn.checkpoint import TTModuleSetEntry, save_module_set_index, save_tt_module
from qwen3_tn.tt import TTMatrixSpec, tt_svd_matrix


class TinyTokenizer:
    eos_token_id = 15
    pad_token_id = 0

    def __call__(self, text: str, **_: Any):
        return {"input_ids": [1 + ord(char) % 14 for char in text]}

    def pad(self, features, *, padding, return_tensors):
        assert padding and return_tensors == "pt"
        width = max(len(x["input_ids"]) for x in features)
        ids = []
        masks = []
        for item in features:
            n = len(item["input_ids"])
            ids.append(list(item["input_ids"]) + [0] * (width - n))
            masks.append([1] * n + [0] * (width - n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.backbone = nn.Module()
        self.backbone.block = nn.Module()
        self.backbone.block.proj = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 16, bias=False)
        self.config = SimpleNamespace(use_cache=True, vocab_size=16)
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.gc_kwargs = gradient_checkpointing_kwargs

    def forward(self, input_ids, labels=None, **_):
        hidden = torch.tanh(self.backbone.block.proj(self.embed(input_ids)))
        logits = self.head(hidden)
        loss = (
            F.cross_entropy(
                logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100
            )
            if labels is not None
            else None
        )
        return SimpleNamespace(loss=loss, logits=logits)


def make_module_set(
    root: Path,
    model: TinyCausalLM | None = None,
    *,
    module_path: str = "backbone.block.proj",
) -> Path:
    model = model or TinyCausalLM()
    spec = TTMatrixSpec((2, 4), (2, 4), (1, 2, 1))
    cores = tt_svd_matrix(
        model.get_submodule(module_path).weight.detach(), spec, svd_driver=None
    )
    directory = root / "modules" / "projection"
    manifest = save_tt_module(
        directory,
        cores,
        spec,
        module_path=module_path,
        model_path="tiny",
        metadata={
            "token_chunk_size": 2,
            "tt_backend": {"name": "native", "version": "2"},
        },
    )
    save_module_set_index(
        root,
        [
            TTModuleSetEntry(
                module_path, "modules/projection", manifest.cores_sha256 or ""
            )
        ],
        model_path="tiny",
    )
    return root / "index.json"


def fake_evaluator(model, config, *, tokenizer):
    del tokenizer
    scale = float(
        sum(parameter.detach().float().sum() for parameter in model.parameters())
    )
    value = abs(scale) % 10 + 1
    return {
        "timing": {},
        "raw_results": {
            "results": {config.task: {metric.name: value for metric in config.metrics}}
        },
    }
