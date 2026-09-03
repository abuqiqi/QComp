"""Small, shared helpers for loading local Hugging Face causal language models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_torch_dtype(value: str) -> torch.dtype:
    try:
        return _TORCH_DTYPES[value]
    except KeyError as error:
        raise ValueError(
            f"unsupported dtype {value!r}; choose from {sorted(_TORCH_DTYPES)}"
        ) from error


def load_local_tokenizer(
    model_path: str | Path, *, ensure_padding: bool = False
) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if ensure_padding and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_local_causal_lm(
    model_path: str | Path,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
    device_map: Any | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM

    options: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        options["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **options)
    return model.to(device) if device is not None else model
