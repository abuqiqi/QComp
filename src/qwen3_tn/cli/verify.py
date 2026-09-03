"""Offline reload and generation check for a saved TT module set."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..model import install_tt_modules
from ..model_loading import load_local_causal_lm, load_local_tokenizer
from ..provenance import atomic_write_json


def verify_saved_tt_model(
    model_path: Path,
    module_set: Path,
    output: Path,
    *,
    prompt: str,
    max_new_tokens: int = 16,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("offline verification requires CUDA")
    started = time.monotonic()
    device_index = 0
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    tokenizer = load_local_tokenizer(model_path, ensure_padding=True)
    model = load_local_causal_lm(
        model_path,
        dtype=torch.bfloat16,
        device=device,
    )
    modules = install_tt_modules(
        model,
        module_set,
        tt_backend="native",
        trainable=False,
        core_dtype=torch.bfloat16,
    )
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    result: dict[str, object] = {
        "format": "qwen3-tn-offline-verification-v1",
        "success": True,
        "module_count": len(modules),
        "prompt": prompt,
        "generated_text": text,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "module_set": str(module_set),
    }
    atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--module-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="请用两句话解释张量网络压缩。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    verify_saved_tt_model(
        args.model_path,
        args.module_set,
        args.output,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
