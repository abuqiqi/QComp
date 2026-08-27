"""CLI for generic TT rank sweeps."""

from __future__ import annotations
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..evaluation import EvaluationConfig, evaluate_causal_lm, extract_metrics
from ..model import TTTarget
from ..workflows import RankSweepCandidate, RankSweepConfig, run_rank_sweep
from .common import check_device, config_argument, parse_dtype, strict_config


def main() -> None:
    args = config_argument(__doc__)
    value = strict_config(
        args.config,
        allowed={
            "model_path",
            "targets",
            "candidates",
            "output_root",
            "cache_root",
            "result_path",
            "evaluation",
            "backend",
            "device",
            "model_dtype",
            "core_dtype",
            "svd_driver",
            "selection_metric",
            "selection_direction",
        },
        required={
            "model_path",
            "targets",
            "candidates",
            "output_root",
            "cache_root",
            "result_path",
            "evaluation",
        },
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
    evaluation = EvaluationConfig.from_json(value["evaluation"])
    targets = tuple(TTTarget.from_dict(item) for item in value["targets"])
    candidates = tuple(
        RankSweepCandidate(
            item["name"], {path: tuple(ranks) for path, ranks in item["ranks"].items()}
        )
        for item in value["candidates"]
    )
    config = RankSweepConfig(
        str(Path(value["model_path"]).resolve()),
        Path(value["output_root"]),
        Path(value["cache_root"]),
        Path(value["result_path"]),
        candidates,
        value.get("backend", "native"),
        parse_dtype(value.get("core_dtype", "bfloat16")),
        value.get("svd_driver", "gesvdj"),
        value.get("selection_metric"),
        value.get("selection_direction", "lower"),
    )

    def evaluator(candidate_model):
        return extract_metrics(
            evaluate_causal_lm(candidate_model, evaluation, tokenizer=tokenizer),
            evaluation,
        )

    print(json.dumps(run_rank_sweep(model, targets, config, evaluator), indent=2))
