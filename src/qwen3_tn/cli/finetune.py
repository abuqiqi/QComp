"""CLI for a generic TT-only causal-LM fine-tuning experiment."""

from __future__ import annotations
import json
from pathlib import Path
from ..data import HFDatasetDocumentSource, JsonlDocumentSource
from ..evaluation import EvaluationConfig
from ..training import TTFineTuneConfig
from ..workflows import FineTuneExperimentConfig, run_finetune_experiment
from .common import config_argument, strict_config


def main() -> None:
    args = config_argument(__doc__)
    value = strict_config(
        args.config,
        allowed={
            "model_path",
            "module_set",
            "artifact_root",
            "result_root",
            "backend",
            "training",
            "evaluation",
            "data",
            "force_retrain",
            "force_reevaluate",
        },
        required={
            "model_path",
            "module_set",
            "artifact_root",
            "result_root",
            "training",
            "data",
        },
    )
    data = value["data"]
    source_type = data.get("type")
    if source_type == "jsonl":
        source = JsonlDocumentSource(
            data["path"], text_field=data.get("text_field", "text")
        )
    elif source_type == "huggingface":
        source = HFDatasetDocumentSource(
            data["dataset"],
            config=data.get("config"),
            split=data.get("split", "train"),
            text_field=data.get("text_field", "text"),
            load_kwargs=data.get("load_kwargs"),
        )
    else:
        raise ValueError(
            "CLI data.type must be jsonl or huggingface; custom loaders use the library API"
        )
    evaluation = (
        EvaluationConfig.from_json(value["evaluation"])
        if value.get("evaluation")
        else None
    )
    config = FineTuneExperimentConfig(
        Path(value["model_path"]),
        Path(value["module_set"]),
        Path(value["artifact_root"]),
        Path(value["result_root"]),
        TTFineTuneConfig(**value["training"]),
        evaluation,
        value.get("backend", "native"),
        bool(value.get("force_retrain", False)),
        bool(value.get("force_reevaluate", False)),
    )
    print(json.dumps(run_finetune_experiment(config, source), indent=2))
