"""CLI for architecture-neutral TT decomposition."""

from __future__ import annotations
import json
from pathlib import Path
from transformers import AutoModelForCausalLM
from ..model import TTTarget
from ..workflows import DecomposeConfig, decompose_targets
from .common import check_device, config_argument, parse_dtype, strict_config


def main() -> None:
    args = config_argument(__doc__)
    value = strict_config(
        args.config,
        allowed={
            "model_path",
            "output_root",
            "cache_root",
            "targets",
            "device",
            "model_dtype",
            "compute_dtype",
            "core_dtype",
            "svd_driver",
            "purpose",
        },
        required={"model_path", "output_root", "cache_root", "targets"},
    )
    device = check_device(value.get("device", "cuda:0"))
    model_dtype = parse_dtype(value.get("model_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        value["model_path"],
        torch_dtype=model_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    targets = tuple(TTTarget.from_dict(item) for item in value["targets"])
    config = DecomposeConfig(
        str(Path(value["model_path"]).expanduser().resolve()),
        Path(value["output_root"]),
        Path(value["cache_root"]),
        value.get("svd_driver", "gesvdj"),
        parse_dtype(value.get("compute_dtype", "float32")),
        parse_dtype(value.get("core_dtype", "bfloat16")),
        value.get("purpose", "tt-decomposition"),
    )
    print(json.dumps(decompose_targets(model, targets, config), indent=2))
