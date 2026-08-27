"""CLI for isolated layer and full causal-LM training benchmarks."""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from ..benchmarking import (
    LayerBenchmarkConfig,
    TrainingBenchmarkConfig,
    benchmark_tt_layer,
    benchmark_tt_training,
)
from ..provenance import load_json
from .common import parse_dtype


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("layer", "training"):
        child = commands.add_parser(name)
        child.add_argument("config", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    value = load_json(args.config)
    output = value.pop("output", None)
    if args.command == "layer":
        value["module_set"] = Path(value["module_set"])
        value["backends"] = tuple(value.get("backends", ("native", "tensorly_torch")))
        value["training_dtype"] = parse_dtype(value.get("training_dtype", "float32"))
        benchmark_tt_layer(LayerBenchmarkConfig(**value), output_path=output)
        return
    model_path = Path(value.pop("model_path"))
    sequence_length = int(value.pop("sequence_length", 1024))
    batch_size = int(value.pop("batch_size", 1))
    value["module_set"] = Path(value["module_set"])
    value["backends"] = tuple(value.get("backends", ("native", "tensorly_torch")))
    config = TrainingBenchmarkConfig(**value)

    def model_factory():
        dtype = (
            torch.bfloat16
            if torch.device(config.device).type == "cuda"
            else torch.float32
        )
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        )

    def batch_factory(model, device):
        vocab = int(model.config.vocab_size)
        ids = torch.randint(
            0,
            vocab,
            (batch_size, sequence_length),
            generator=torch.Generator().manual_seed(42),
        ).to(device)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }

    benchmark_tt_training(config, model_factory, batch_factory, output_path=output)
