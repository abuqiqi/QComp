"""CLI for dense or TT causal-LM evaluation."""

from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..evaluation import EvaluationConfig, evaluate_causal_lm
from ..model import TTModulePatch
from .common import check_device, config_argument, parse_dtype, strict_config


def main() -> None:
    args = config_argument(__doc__)
    value = strict_config(
        args.config,
        allowed={
            "model_path",
            "evaluation",
            "module_set",
            "backend",
            "output",
            "device",
            "model_dtype",
        },
        required={"model_path", "evaluation", "output"},
    )
    device = check_device(value.get("device", "cuda:0"))
    tokenizer = AutoTokenizer.from_pretrained(
        value["model_path"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        value["model_path"],
        torch_dtype=parse_dtype(value.get("model_dtype", "bfloat16")),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    config = EvaluationConfig.from_json(value["evaluation"])
    if value.get("module_set"):
        with TTModulePatch(
            model,
            value["module_set"],
            tt_backend=value.get("backend", "native"),
            core_dtype=parse_dtype(value.get("model_dtype", "bfloat16")),
        ):
            evaluate_causal_lm(
                model, config, tokenizer=tokenizer, output_path=value["output"]
            )
    else:
        evaluate_causal_lm(
            model, config, tokenizer=tokenizer, output_path=value["output"]
        )
