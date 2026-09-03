"""Architecture-neutral, cached TT decomposition workflow."""

from __future__ import annotations
import hashlib
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import torch
from torch import nn
from ..checkpoint import (
    TTModuleSetEntry,
    load_tt_module,
    publish_module_set,
    save_module_set_index,
    save_tt_module,
)
from ..model import TTTarget
from ..provenance import canonical_json_sha256, file_sha256
from ..tt import reconstruct_matrix, tt_svd_matrix


@dataclass(frozen=True)
class DecomposeConfig:
    model_path: str
    output_root: Path
    cache_root: Path
    svd_driver: str | None = "gesvdj"
    compute_dtype: torch.dtype = torch.float32
    core_dtype: torch.dtype = torch.bfloat16
    purpose: str = "tt-decomposition"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def decompose_targets(
    model: nn.Module, targets: Sequence[TTTarget], config: DecomposeConfig
) -> dict[str, Any]:
    if not targets:
        raise ValueError("targets must not be empty")
    paths = [target.module_path for target in targets]
    if len(set(paths)) != len(paths):
        raise ValueError("target module paths must be unique")
    output = Path(config.output_root)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    entries: list[TTModuleSetEntry] = []
    reports: dict[str, Any] = {}
    try:
        for target in targets:
            try:
                module = model.get_submodule(target.module_path)
            except AttributeError as error:
                raise KeyError(
                    f"module path does not exist: {target.module_path}"
                ) from error
            if not isinstance(module, nn.Linear) or module.bias is not None:
                raise TypeError(
                    f"{target.module_path} must be bias-free torch.nn.Linear"
                )
            if tuple(module.weight.shape) != (
                target.spec.out_features,
                target.spec.in_features,
            ):
                raise ValueError(
                    f"weight shape does not match TT spec for {target.module_path}"
                )
            weight_sha = _tensor_sha256(module.weight)
            descriptor = {
                "format": "qwen3-tn-decomposition-input-v1",
                "model_path": config.model_path,
                "module_path": target.module_path,
                "weight_sha256": weight_sha,
                "spec": target.spec.to_dict(),
                "svd_driver": config.svd_driver,
                "compute_dtype": str(config.compute_dtype),
                "core_dtype": str(config.core_dtype),
            }
            key = canonical_json_sha256(descriptor)
            cache = Path(config.cache_root) / key
            source = "cache"
            try:
                cores, cached_manifest = load_tt_module(cache)
                if cached_manifest.metadata.get("decomposition_input") != descriptor:
                    raise ValueError("cache descriptor mismatch")
            except (FileNotFoundError, ValueError, KeyError):
                source = "decomposition"
                cores = tt_svd_matrix(
                    module.weight.detach().to(config.compute_dtype),
                    target.spec,
                    svd_driver=config.svd_driver,
                )
                cores = [core.to(config.core_dtype) for core in cores]
                save_tt_module(
                    cache,
                    cores,
                    target.spec,
                    module_path=target.module_path,
                    model_path=config.model_path,
                    purpose="content-addressed-decomposition-cache",
                    metadata={
                        "decomposition_input": descriptor,
                        "token_chunk_size": target.token_chunk_size,
                    },
                )
            reference = module.weight.detach().to(config.compute_dtype)
            reconstructed = reconstruct_matrix(
                [
                    core.to(device=reference.device, dtype=config.compute_dtype)
                    for core in cores
                ],
                target.spec,
            )
            relative_error = float(
                torch.linalg.vector_norm(reconstructed - reference)
                / torch.linalg.vector_norm(reference)
            )
            relative = Path("modules") / target.module_path.replace(".", "__")
            manifest = save_tt_module(
                temporary / relative,
                cores,
                target.spec,
                module_path=target.module_path,
                model_path=config.model_path,
                purpose=config.purpose,
                metrics={"weight_relative_l2": relative_error},
                metadata={
                    "token_chunk_size": target.token_chunk_size,
                    "decomposition_input": descriptor,
                },
            )
            entries.append(
                TTModuleSetEntry(
                    target.module_path, relative.as_posix(), str(manifest.cores_sha256)
                )
            )
            reports[target.module_path] = {
                "source": source,
                "cache_key": key,
                "weight_relative_l2": relative_error,
                "dense_parameters": target.spec.dense_num_parameters,
                "tt_parameters": target.spec.num_parameters,
                "compression_ratio": target.spec.compression_ratio,
            }
        save_module_set_index(
            temporary,
            entries,
            model_path=config.model_path,
            purpose=config.purpose,
            metadata={"targets": [target.to_dict() for target in targets]},
        )
        publish_module_set(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    dense = sum(target.spec.dense_num_parameters for target in targets)
    tt = sum(target.spec.num_parameters for target in targets)
    return {
        "format": "qwen3-tn-decomposition-result-v1",
        "module_set": str(output / "index.json"),
        "targets": reports,
        "aggregate": {
            "target_count": len(targets),
            "dense_parameters": dense,
            "tt_parameters": tt,
            "compression_ratio": dense / tt,
        },
    }
