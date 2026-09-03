"""CLI for dense or TT causal-LM evaluation."""

from __future__ import annotations
from ..evaluation import EvaluationConfig, evaluate_causal_lm
from ..model_loading import load_local_causal_lm, load_local_tokenizer
from ..model import TTModulePatch
from .common import (
    check_device,
    config_argument,
    load_backend_modules,
    parse_backend_options,
    parse_dtype,
    strict_config,
)


def main() -> None:
    args = config_argument(__doc__)
    value = strict_config(
        args.config,
        allowed={
            "model_path",
            "evaluation",
            "module_set",
            "backend",
            "backend_modules",
            "backend_options",
            "output",
            "device",
            "model_dtype",
        },
        required={"model_path", "evaluation", "output"},
    )
    load_backend_modules(value.get("backend_modules"))
    backend_options = parse_backend_options(value.get("backend_options"))
    device = check_device(value.get("device", "cuda:0"))
    tokenizer = load_local_tokenizer(value["model_path"])
    model = load_local_causal_lm(
        value["model_path"],
        dtype=parse_dtype(value.get("model_dtype", "bfloat16")),
        device=device,
    )
    config = EvaluationConfig.from_json(value["evaluation"])
    if value.get("module_set"):
        with TTModulePatch(
            model,
            value["module_set"],
            tt_backend=value.get("backend", "native"),
            core_dtype=parse_dtype(value.get("model_dtype", "bfloat16")),
            backend_options=backend_options,
        ):
            evaluate_causal_lm(
                model, config, tokenizer=tokenizer, output_path=value["output"]
            )
    else:
        evaluate_causal_lm(
            model, config, tokenizer=tokenizer, output_path=value["output"]
        )


if __name__ == "__main__":
    main()
