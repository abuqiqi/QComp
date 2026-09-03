"""CLI for a generic TT-only causal-LM fine-tuning experiment."""

from __future__ import annotations
import json
from pathlib import Path
from ..data import HFDatasetDocumentSource, JsonlDocumentSource
from ..evaluation import EvaluationConfig
from ..training import TTFineTuneConfig
from ..workflows import FineTuneExperimentConfig, run_finetune_experiment
from .common import (
    config_argument,
    load_backend_modules,
    parse_backend_options,
    strict_config,
)


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
            "backend_modules",
            "backend_options",
            "training",
            "evaluation",
            "data",
            "force_retrain",
            "force_reevaluate",
            "resume_from",
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
    load_backend_modules(value.get("backend_modules"))
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
        model_path=Path(value["model_path"]),
        module_set=Path(value["module_set"]),
        artifact_root=Path(value["artifact_root"]),
        result_root=Path(value["result_root"]),
        training=TTFineTuneConfig(**value["training"]),
        evaluation=evaluation,
        tt_backend=value.get("backend", "native"),
        force_retrain=bool(value.get("force_retrain", False)),
        force_reevaluate=bool(value.get("force_reevaluate", False)),
        resume_from=value.get("resume_from"),
        backend_options=parse_backend_options(value.get("backend_options")),
    )
    print(json.dumps(run_finetune_experiment(config, source), indent=2))


if __name__ == "__main__":
    main()
