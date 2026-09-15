"""缓存未压缩模型的局部输入，并编排单投影压缩候选的输出 NMSE 评测。

本模块把校准 DataLoader 送入未压缩模型，将目标 Linear 的输入与有效位置 mask 保留在
模型设备或分片保存到磁盘，随后只运行原始和压缩后的目标 Linear。它复用通用压缩计划、
后端和评测结果，不负责 Qwen3 目标选择、候选 rank 策略或实验报告。

主要内容：
- ``local_output_cache_fingerprint``、``resolve_local_output_cache_path``：确定缓存身份与位置。
- ``capture_local_output_inputs``：执行 dense reference 并原子保存输入分片。
- ``capture_local_output_inputs_in_memory``：在模型设备临时保存输入并复用共享 storage。
- ``validate_local_output_input_cache``、``iter_local_output_input_batches``：验证并流式读取缓存。
- ``evaluate_local_output_nmse_plans``、``evaluate_local_output_nmse_plans_in_memory``：分别
  从磁盘和内存输入计算局部输出 NMSE。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..backends import TensorNetworkBackend
from ..evaluation import EvaluationResult, EvaluationTask
from ..evaluation.local_output_nmse import local_output_error_sums, local_output_nmse
from ..model import find_linear, list_tensor_network_linears
from .compress import CompressionPlan
from .evaluate import (
    CompressionEvaluationResult,
    CompressionPlanEvaluation,
    TimedEvaluation,
    evaluate_compression_plans,
)

CACHE_FORMAT_VERSION = 1
MemoryInputs = dict[tuple[str, str], list[Tensor]]
MemoryMasks = dict[str, list[Tensor]]
SourceRecords = Mapping[str, Mapping[str, int | float]]
BatchIterator = Callable[[str, str], Iterable[tuple[Tensor, Tensor]]]


def _canonical_json(value: object) -> str:
    """将缓存身份字段转换为稳定、紧凑的 JSON 文本。

    参数：
        value: 待序列化的 JSON 兼容对象。

    返回：
        字典键排序且没有多余空白的 JSON 文本。
    """
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("cache identity must be JSON serializable") from error


def _validated_module_paths(module_paths: Sequence[str]) -> tuple[str, ...]:
    """规范并校验有序且不重复的模块路径。

    参数：
        module_paths: 外部传入的模块路径序列。

    返回：
        保持输入顺序的模块路径元组。
    """
    paths = tuple(module_paths)
    if not paths or any(
        not isinstance(path, str) or not path.strip() or path != path.strip()
        for path in paths
    ):
        raise ValueError("module_paths must contain non-empty normalized paths")
    if len(set(paths)) != len(paths):
        raise ValueError("module_paths must not contain duplicates")
    return paths


def local_output_cache_fingerprint(
    *,
    model_identity: Mapping[str, object],
    calibration_config: Mapping[str, object],
    module_paths: Sequence[str],
) -> str:
    """计算模型、校准配置和目标模块共同决定的 baseline input 缓存标识。

    参数：
        model_identity: 可序列化的模型及 tokenizer 身份字段。
        calibration_config: 可序列化的完整校准数据配置。
        module_paths: 要捕获输入的有序模块路径。

    返回：
        六十四位 SHA-256 十六进制字符串。
    """
    paths = _validated_module_paths(module_paths)
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_identity": dict(model_identity),
        "calibration_config": dict(calibration_config),
        "module_paths": list(paths),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_local_output_cache_path(
    cache_root: Path,
    *,
    model_identity: Mapping[str, object],
    calibration_config: Mapping[str, object],
    module_paths: Sequence[str],
) -> Path:
    """返回给定实验身份对应的默认 baseline input 缓存目录。

    参数：
        cache_root: 通常为 ``ArtifactPaths.cache`` 的统一缓存根目录。
        model_identity: 模型及 tokenizer 身份字段。
        calibration_config: 完整校准数据配置。
        module_paths: 要捕获输入的有序模块路径。

    返回：
        尚未创建的或已经存在的确定性缓存路径。
    """
    fingerprint = local_output_cache_fingerprint(
        model_identity=model_identity,
        calibration_config=calibration_config,
        module_paths=module_paths,
    )
    return Path(cache_root).expanduser() / "local-output-inputs" / fingerprint


def _model_device(model: nn.Module) -> torch.device:
    """返回模型首个参数所在设备，无参数模型使用 CPU。

    参数：
        model: 待执行 reference forward 的模型。

    返回：
        首个参数的设备或 CPU 设备。
    """
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _storage_key(value: Tensor) -> tuple[object, ...]:
    """返回足以识别同一张量视图底层 storage 的运行时键。"""
    return (
        value.device.type,
        value.device.index,
        value.untyped_storage().data_ptr(),
        value.storage_offset(),
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
    )


def capture_local_output_inputs_in_memory(
    model: nn.Module,
    dataloaders: Mapping[str, Iterable[Mapping[str, Tensor]]],
    module_paths: Sequence[str],
) -> tuple[MemoryInputs, MemoryMasks, dict[str, dict[str, int | float]]]:
    """遍历 dense 校准集并在模型设备临时保存目标 Linear 输入。

    参数：
        model: 尚未安装压缩层的 reference 模型。
        dataloaders: 校准来源名称到可重复迭代 batch 的非空映射。
        module_paths: 当前批次需要捕获输入的有序 Linear 路径。

    返回：
        ``(inputs, masks, source_records)``。inputs 使用 ``(source, module_path)``
        作为键；同一 batch 中共享 storage 的模块引用同一份 clone。
    """
    paths = _validated_module_paths(module_paths)
    if not dataloaders:
        raise ValueError("dataloaders must not be empty")
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in dataloaders
    ):
        raise ValueError("dataloader names must be non-empty normalized strings")
    linears = {path: find_linear(model, path) for path in paths}
    inputs: MemoryInputs = {
        (source, path): [] for source in dataloaders for path in paths
    }
    masks: MemoryMasks = {source: [] for source in dataloaders}
    source_records: dict[str, dict[str, int | float]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    captured: dict[str, Tensor] = {}
    shared: dict[tuple[object, ...], Tensor] = {}
    training = [(module, module.training) for module in model.modules()]
    device = _model_device(model)

    def capture(path: str) -> Callable[[nn.Module, tuple[object, ...]], None]:
        """创建当前模块的设备内输入捕获 hook。"""

        def hook(module: nn.Module, positional: tuple[object, ...]) -> None:
            """校验输入，并让共享 storage 的投影复用同一份安全 clone。"""
            del module
            if path in captured:
                raise ValueError(
                    f"target module executed more than once in a batch: {path}"
                )
            if (
                not positional
                or not isinstance(positional[0], Tensor)
                or positional[0].ndim != 3
            ):
                raise ValueError(f"target input must be a rank-3 tensor: {path}")
            value = positional[0]
            key = _storage_key(value)
            if key not in shared:
                shared[key] = value.detach().clone()
            captured[path] = shared[key]

        return hook

    try:
        for path, linear in linears.items():
            handles.append(linear.register_forward_pre_hook(capture(path)))
        model.eval()
        with torch.inference_mode():
            for source, dataloader in dataloaders.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                batches = examples = valid_tokens = tensor_bytes = 0
                for batch in dataloader:
                    if not isinstance(batch, Mapping):
                        raise ValueError(
                            f"calibration batch must be a mapping: {source}"
                        )
                    input_ids = batch.get("input_ids")
                    attention_mask = batch.get("attention_mask")
                    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                        raise ValueError("input_ids must have shape [batch, sequence]")
                    if (
                        not isinstance(attention_mask, Tensor)
                        or attention_mask.shape != input_ids.shape
                    ):
                        raise ValueError("attention_mask must match input_ids")
                    captured.clear()
                    shared.clear()
                    device_input_ids = input_ids.to(device)
                    device_mask = attention_mask.to(device)
                    output = model(
                        input_ids=device_input_ids,
                        attention_mask=device_mask,
                        use_cache=False,
                    )
                    del output, device_input_ids
                    missing = set(paths) - set(captured)
                    if missing:
                        raise ValueError(
                            f"target modules were not executed: {', '.join(sorted(missing))}"
                        )
                    saved_mask = device_mask.detach().clone()
                    masks[source].append(saved_mask)
                    tensor_bytes += saved_mask.numel() * saved_mask.element_size()
                    tensor_bytes += sum(
                        value.numel() * value.element_size()
                        for value in shared.values()
                    )
                    for path in paths:
                        inputs[(source, path)].append(captured[path])
                    batches += 1
                    examples += int(input_ids.shape[0])
                    valid_tokens += int(
                        attention_mask.to(dtype=torch.bool).sum().item()
                    )
                    captured.clear()
                    shared.clear()
                if batches == 0:
                    raise ValueError(
                        f"calibration source produced no batches: {source}"
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                source_records[source] = {
                    "batches": batches,
                    "evaluated_examples": examples,
                    "valid_tokens": valid_tokens,
                    "capture_seconds": time.perf_counter() - started,
                    "input_tensor_bytes": tensor_bytes,
                }
        return inputs, masks, source_records
    finally:
        captured.clear()
        shared.clear()
        for handle in handles:
            handle.remove()
        for module, was_training in training:
            module.training = was_training


def capture_local_output_inputs(
    model: nn.Module,
    dataloaders: Mapping[str, Iterable[Mapping[str, Tensor]]],
    module_paths: Sequence[str],
    destination: Path,
    *,
    model_identity: Mapping[str, object],
    calibration_config: Mapping[str, object],
) -> Path:
    """遍历一次 dense 校准集，并原子保存每个目标 Linear 的输入与 mask。

    参数：
        model: 尚未安装压缩层的 reference 模型。
        dataloaders: 校准来源名称到 batch iterable 的非空映射。
        module_paths: 本次需要捕获输入的有序 Linear 路径。
        destination: 必须尚不存在的目标缓存目录。
        model_identity: 写入 manifest 的模型及 tokenizer 身份字段。
        calibration_config: 写入 manifest 的完整校准配置。

    返回：
        发布成功的 ``manifest.json`` 路径。

    异常：
        FileExistsError: destination 已存在时抛出。
        ValueError: 来源、batch、mask 或 hook 捕获结果不符合约定时抛出。
    """
    paths = _validated_module_paths(module_paths)
    if not dataloaders:
        raise ValueError("dataloaders must not be empty")
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in dataloaders
    ):
        raise ValueError("dataloader names must be non-empty normalized strings")
    destination = Path(destination).expanduser()
    if destination.exists():
        raise FileExistsError(f"baseline input cache already exists: {destination}")
    linears = {path: find_linear(model, path) for path in paths}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    handles: list[torch.utils.hooks.RemovableHandle] = []
    captured: dict[str, Tensor] = {}
    training = [(module, module.training) for module in model.modules()]
    device = _model_device(model)
    capture_started = time.perf_counter()

    def capture(path: str) -> Callable[[nn.Module, tuple[object, ...]], None]:
        """创建一个只捕获当前 batch 第一个位置输入的 pre-hook。

        参数：
            path: hook 对应的完整模块路径。

        返回：
            可注册到 Linear 的 forward pre-hook。
        """

        def hook(module: nn.Module, inputs: tuple[object, ...]) -> None:
            """将三维 Linear 输入复制到 CPU 暂存。

            参数：
                module: 当前触发 hook 的 Linear；路径已由外层闭包记录。
                inputs: 当前 Linear 的位置参数元组。
            """
            del module
            if path in captured:
                raise ValueError(
                    f"target module executed more than once in a batch: {path}"
                )
            if not inputs or not isinstance(inputs[0], Tensor) or inputs[0].ndim != 3:
                raise ValueError(f"target input must be a rank-3 tensor: {path}")
            captured[path] = inputs[0].detach().to(device="cpu")

        return hook

    try:
        for path, linear in linears.items():
            handles.append(linear.register_forward_pre_hook(capture(path)))
        (temporary / "masks").mkdir()
        for index in range(len(paths)):
            (temporary / "inputs" / f"{index:04d}").mkdir(parents=True)

        source_records: dict[str, dict[str, object]] = {}
        module_records = {
            path: {"index": index, "input_shape": None, "dtype": None}
            for index, path in enumerate(paths)
        }
        model.eval()
        with torch.inference_mode():
            for source_index, (source, dataloader) in enumerate(dataloaders.items()):
                source_started = time.perf_counter()
                batches = examples = valid_tokens = 0
                for batch_index, batch in enumerate(dataloader):
                    if not isinstance(batch, Mapping):
                        raise ValueError(
                            f"calibration batch must be a mapping: {source}"
                        )
                    input_ids, attention_mask = batch.get("input_ids"), batch.get(
                        "attention_mask"
                    )
                    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                        raise ValueError("input_ids must have shape [batch, sequence]")
                    if (
                        not isinstance(attention_mask, Tensor)
                        or attention_mask.shape != input_ids.shape
                    ):
                        raise ValueError("attention_mask must match input_ids")
                    captured.clear()
                    output = model(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        use_cache=False,
                    )
                    del output
                    missing = set(paths) - set(captured)
                    if missing:
                        raise ValueError(
                            f"target modules were not executed: {', '.join(sorted(missing))}"
                        )
                    stem = f"{source_index:03d}-{batch_index:05d}.pt"
                    mask = attention_mask.detach().to(device="cpu")
                    torch.save(mask, temporary / "masks" / stem)
                    for module_index, path in enumerate(paths):
                        value = captured[path]
                        record = module_records[path]
                        shape, dtype = list(value.shape[1:]), str(value.dtype)
                        if record["input_shape"] is None:
                            record.update(input_shape=shape, dtype=dtype)
                        elif record["input_shape"] != shape or record["dtype"] != dtype:
                            raise ValueError(
                                f"input metadata changed across batches: {path}"
                            )
                        torch.save(
                            value, temporary / "inputs" / f"{module_index:04d}" / stem
                        )
                    batch_examples = int(input_ids.shape[0])
                    batches += 1
                    examples += batch_examples
                    valid_tokens += int(
                        attention_mask.to(dtype=torch.bool).sum().item()
                    )
                    captured.clear()
                if batches == 0:
                    raise ValueError(
                        f"calibration source produced no batches: {source}"
                    )
                source_records[source] = {
                    "index": source_index,
                    "batches": batches,
                    "evaluated_examples": examples,
                    "valid_tokens": valid_tokens,
                    "capture_seconds": time.perf_counter() - source_started,
                }

        fingerprint = local_output_cache_fingerprint(
            model_identity=model_identity,
            calibration_config=calibration_config,
            module_paths=paths,
        )
        manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "model_identity": dict(model_identity),
            "calibration_config": dict(calibration_config),
            "module_paths": list(paths),
            "modules": module_records,
            "sources": source_records,
            "capture_seconds": time.perf_counter() - capture_started,
            "cache_tensor_bytes": sum(
                path.stat().st_size for path in temporary.rglob("*.pt")
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.rename(temporary, destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        captured.clear()
        for handle in handles:
            handle.remove()
        for module, was_training in training:
            module.training = was_training


def validate_local_output_input_cache(
    cache_path: Path,
    *,
    expected_model_identity: Mapping[str, object],
    expected_calibration_config: Mapping[str, object],
    required_module_paths: Sequence[str],
) -> Mapping[str, object]:
    """加载 baseline input manifest，并验证身份和全部张量分片。

    参数：
        cache_path: 包含 ``manifest.json`` 的缓存目录。
        expected_model_identity: 本次模型及 tokenizer 身份字段。
        expected_calibration_config: 本次完整校准配置。
        required_module_paths: 本次必须存在的有序模块路径。

    返回：
        验证通过的 manifest 映射。

    异常：
        FileNotFoundError: 缓存目录、manifest 或分片缺失时抛出。
        ValueError: schema 或实验身份不匹配时抛出。
    """
    paths = _validated_module_paths(required_module_paths)
    root = Path(cache_path).expanduser()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"baseline input manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("baseline input manifest is not valid JSON") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != CACHE_FORMAT_VERSION
    ):
        raise ValueError("unsupported baseline input cache format")
    expected_fingerprint = local_output_cache_fingerprint(
        model_identity=expected_model_identity,
        calibration_config=expected_calibration_config,
        module_paths=paths,
    )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("baseline input cache identity does not match this experiment")
    if manifest.get("module_paths") != list(paths):
        raise ValueError("baseline input cache module paths do not match")
    modules, sources = manifest.get("modules"), manifest.get("sources")
    if not isinstance(modules, dict) or not isinstance(sources, dict) or not sources:
        raise ValueError("baseline input manifest has invalid modules or sources")
    capture_seconds = manifest.get("capture_seconds")
    cache_tensor_bytes = manifest.get("cache_tensor_bytes")
    if (
        not isinstance(capture_seconds, int | float)
        or not math.isfinite(float(capture_seconds))
        or capture_seconds < 0
        or type(cache_tensor_bytes) is not int
        or cache_tensor_bytes <= 0
    ):
        raise ValueError("baseline input cache size or capture time is invalid")
    module_indices = set()
    for path in paths:
        record = modules.get(path)
        if not isinstance(record, dict) or type(record.get("index")) is not int:
            raise ValueError(f"baseline module metadata is invalid: {path}")
        shape, dtype = record.get("input_shape"), record.get("dtype")
        if (
            record["index"] < 0
            or record["index"] in module_indices
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ValueError(f"baseline module metadata is invalid: {path}")
        module_indices.add(record["index"])
    source_indices = set()
    for source in sources.values():
        if not isinstance(source, dict):
            raise ValueError("baseline input source metadata is invalid")
        source_index, batch_count = source.get("index"), source.get("batches")
        if (
            type(source_index) is not int
            or type(batch_count) is not int
            or batch_count <= 0
            or source_index < 0
            or source_index in source_indices
        ):
            raise ValueError("baseline input source indices are invalid")
        source_indices.add(source_index)
        if (
            type(source.get("evaluated_examples")) is not int
            or source["evaluated_examples"] <= 0
            or type(source.get("valid_tokens")) is not int
            or source["valid_tokens"] <= 0
            or not isinstance(source.get("capture_seconds"), int | float)
            or not math.isfinite(float(source["capture_seconds"]))
            or source["capture_seconds"] < 0
        ):
            raise ValueError("baseline input source counts are invalid")
        for batch_index in range(batch_count):
            stem = f"{source_index:03d}-{batch_index:05d}.pt"
            if not (root / "masks" / stem).is_file():
                raise FileNotFoundError(f"baseline mask shard is missing: {stem}")
            for path in paths:
                record = modules.get(path)
                if not isinstance(record, dict) or type(record.get("index")) is not int:
                    raise ValueError(f"baseline module metadata is invalid: {path}")
                shard = root / "inputs" / f"{record['index']:04d}" / stem
                if not shard.is_file():
                    raise FileNotFoundError(f"baseline input shard is missing: {shard}")
    return manifest


def iter_local_output_input_batches(
    cache_path: Path,
    manifest: Mapping[str, object],
    *,
    source: str,
    module_path: str,
) -> Iterator[tuple[Tensor, Tensor]]:
    """按记录顺序流式读取一个模块和校准来源的输入及有效位置 mask。

    参数：
        cache_path: 已验证的 baseline input 缓存目录。
        manifest: ``validate_local_output_input_cache`` 返回的 manifest。
        source: manifest 中的校准来源名称。
        module_path: manifest 中的模块路径。

    返回：
        逐 batch 产生 CPU 上的 ``(module_input, valid_mask)``。

    异常：
        KeyError: 来源或模块不存在时抛出。
        ValueError: 加载张量的形状不符合 manifest 时抛出。
    """
    root = Path(cache_path).expanduser()
    source_record = manifest["sources"][source]  # type: ignore[index]
    module_record = manifest["modules"][module_path]  # type: ignore[index]
    source_index = source_record["index"]
    module_index = module_record["index"]
    for batch_index in range(source_record["batches"]):
        stem = f"{source_index:03d}-{batch_index:05d}.pt"
        module_input = torch.load(
            root / "inputs" / f"{module_index:04d}" / stem,
            map_location="cpu",
            weights_only=True,
        )
        valid_mask = torch.load(
            root / "masks" / stem, map_location="cpu", weights_only=True
        )
        if not isinstance(module_input, Tensor) or module_input.ndim != 3:
            raise ValueError(f"cached module input is invalid: {module_path}/{stem}")
        if (
            list(module_input.shape[1:]) != module_record["input_shape"]
            or str(module_input.dtype) != module_record["dtype"]
        ):
            raise ValueError(
                f"cached module input does not match manifest: {module_path}/{stem}"
            )
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.shape != module_input.shape[:2]
        ):
            raise ValueError(f"cached valid mask is invalid: {source}/{stem}")
        yield module_input, valid_mask


def _evaluate_local_output_nmse_plans(
    model: nn.Module,
    plans: Mapping[str, CompressionPlan],
    *,
    available_module_paths: Sequence[str],
    source_records: SourceRecords,
    preprocessing: str,
    iter_batches: BatchIterator,
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    decomposition_dtype: torch.dtype | None,
    on_plan_result: Callable[[CompressionPlanEvaluation], None] | None,
) -> CompressionEvaluationResult:
    """使用统一 batch 迭代入口执行单目标局部输出 NMSE 评测。"""
    if not plans:
        raise ValueError("plans must not be empty")
    targets = []
    for plan in plans.values():
        if len(plan.targets) != 1:
            raise ValueError("local output NMSE plans must contain exactly one target")
        targets.append(plan.targets[0].module_path)
    target_paths = tuple(dict.fromkeys(targets))
    if not set(target_paths).issubset(set(available_module_paths)):
        raise ValueError("local output NMSE targets are missing from the input source")
    originals = {path: find_linear(model, path) for path in target_paths}
    if not source_records:
        raise ValueError("local output input source contains no calibration sources")

    evaluators = {}
    baselines = {}
    for source, source_record in source_records.items():
        if not isinstance(source, str) or not isinstance(source_record, Mapping):
            raise ValueError("local output input source metadata is invalid")
        examples = source_record.get("evaluated_examples")
        tokens = source_record.get("valid_tokens")
        if (
            type(examples) is not int
            or examples <= 0
            or type(tokens) is not int
            or tokens <= 0
        ):
            raise ValueError(f"baseline input counts are invalid: {source}")
        task = EvaluationTask(
            name=f"local-output-nmse:{source}",
            dataset=source,
            split="calibration",
            preprocessing=preprocessing,
            requested_metrics=("output_nmse",),
        )
        baselines[source] = TimedEvaluation(
            EvaluationResult(
                task=task,
                metrics={"output_nmse": 0.0},
                evaluated_examples=examples,
                evaluated_tokens=tokens,
                total_examples=examples,
            ),
            0.0,
        )

        def evaluate(
            current_model: nn.Module,
            *,
            current_source: str = source,
            current_task: EvaluationTask = task,
            expected_examples: int = examples,
            expected_tokens: int = tokens,
        ) -> EvaluationResult:
            """直接运行当前唯一压缩层及其原始 Linear，不执行完整模型。"""
            installed = [
                (path, module)
                for path, module in list_tensor_network_linears(current_model)
                if path in originals
            ]
            if len(installed) != 1:
                raise ValueError(
                    "local output evaluator requires exactly one compressed target"
                )
            path, compressed = installed[0]
            original = originals[path]
            device = original.weight.device
            error_sum = power_sum = 0.0
            valid_tokens = evaluated_examples = 0
            with torch.inference_mode():
                for module_input, valid_mask in iter_batches(current_source, path):
                    value = module_input.to(device=device, dtype=original.weight.dtype)
                    mask = valid_mask.to(device=device)
                    reference_output = original(value)
                    compressed_output = compressed(value)
                    batch_error, batch_power, batch_tokens = local_output_error_sums(
                        reference_output, compressed_output, mask
                    )
                    error_sum += float(batch_error)
                    power_sum += float(batch_power)
                    valid_tokens += batch_tokens
                    evaluated_examples += int(value.shape[0])
            if (
                evaluated_examples != expected_examples
                or valid_tokens != expected_tokens
            ):
                raise ValueError(
                    f"local output batches do not match source metadata: {current_source}"
                )
            return EvaluationResult(
                task=current_task,
                metrics={
                    "output_nmse": local_output_nmse(error_sum, power_sum),
                    "squared_error_sum": error_sum,
                    "target_power_sum": power_sum,
                },
                evaluated_examples=evaluated_examples,
                evaluated_tokens=valid_tokens,
                total_examples=evaluated_examples,
            )

        evaluators[source] = evaluate

    return evaluate_compression_plans(
        model,
        plans,
        evaluators=evaluators,
        metric_directions={
            source: {"output_nmse": "lower"} for source in source_records
        },
        decomposition_backends=decomposition_backends,
        execution_backends=execution_backends,
        baseline=baselines,
        evaluate_baseline=False,
        decomposition_dtype=decomposition_dtype,
        collect_layer_metrics=True,
        on_plan_result=on_plan_result,
    )


def evaluate_local_output_nmse_plans(
    model: nn.Module,
    plans: Mapping[str, CompressionPlan],
    *,
    cache_path: Path,
    manifest: Mapping[str, object],
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    decomposition_dtype: torch.dtype | None = None,
    on_plan_result: Callable[[CompressionPlanEvaluation], None] | None = None,
) -> CompressionEvaluationResult:
    """从已验证磁盘缓存对单目标计划计算各来源的局部输出 NMSE。"""
    cached_paths = manifest.get("module_paths")
    sources = manifest.get("sources")
    if not isinstance(cached_paths, list) or not all(
        isinstance(path, str) for path in cached_paths
    ):
        raise ValueError("baseline input manifest has invalid module paths")
    if not isinstance(sources, Mapping):
        raise ValueError("baseline input manifest contains no calibration sources")
    fingerprint = str(manifest.get("fingerprint", ""))

    def batches(source: str, module_path: str) -> Iterable[tuple[Tensor, Tensor]]:
        """流式读取磁盘缓存中的一个来源和模块。"""
        return iter_local_output_input_batches(
            cache_path, manifest, source=source, module_path=module_path
        )

    return _evaluate_local_output_nmse_plans(
        model,
        plans,
        available_module_paths=cached_paths,
        source_records=sources,  # type: ignore[arg-type]
        preprocessing=f"cached-inputs:{fingerprint}",
        iter_batches=batches,
        decomposition_backends=decomposition_backends,
        execution_backends=execution_backends,
        decomposition_dtype=decomposition_dtype,
        on_plan_result=on_plan_result,
    )


def evaluate_local_output_nmse_plans_in_memory(
    model: nn.Module,
    plans: Mapping[str, CompressionPlan],
    *,
    inputs: Mapping[tuple[str, str], Sequence[Tensor]],
    masks: Mapping[str, Sequence[Tensor]],
    source_records: SourceRecords,
    decomposition_backends: Mapping[str, TensorNetworkBackend[Any]],
    execution_backends: Mapping[str, TensorNetworkBackend[Any]],
    decomposition_dtype: torch.dtype | None = None,
    on_plan_result: Callable[[CompressionPlanEvaluation], None] | None = None,
) -> CompressionEvaluationResult:
    """从模型设备内的临时输入对单目标计划计算局部输出 NMSE。

    参数：
        model: 与内存输入对应、当前未压缩的模型。
        plans: 名称到单个 ``CompressionTarget`` 计划的非空映射。
        inputs: ``(source, module_path)`` 到逐 batch 输入张量的映射。
        masks: 来源名称到逐 batch 有效位置 mask 的映射。
        source_records: 每个来源的 batch、样本、token 和捕获统计。
        decomposition_backends: 各表示的分解后端。
        execution_backends: 各表示的执行后端。
        decomposition_dtype: 可选分解精度。
        on_plan_result: 每个候选恢复原模块后的通知函数。

    返回：
        复用通用类型的 baseline 与逐计划压缩评测结果。
    """
    if not inputs or not masks or set(masks) != set(source_records):
        raise ValueError("in-memory local output inputs have invalid sources")
    available_paths = tuple(dict.fromkeys(path for _, path in inputs))
    for source in source_records:
        for path in available_paths:
            if (source, path) not in inputs:
                raise ValueError(
                    f"in-memory local output input is missing: {source}/{path}"
                )

    def batches(source: str, module_path: str) -> Iterator[tuple[Tensor, Tensor]]:
        """校验并迭代内存中的一个来源和模块。"""
        values = inputs[(source, module_path)]
        valid_masks = masks[source]
        if not values or len(values) != len(valid_masks):
            raise ValueError(
                f"in-memory local output batch count is invalid: {source}/{module_path}"
            )
        for value, mask in zip(values, valid_masks, strict=True):
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise ValueError(
                    f"in-memory module input is invalid: {source}/{module_path}"
                )
            if not isinstance(mask, Tensor) or mask.shape != value.shape[:2]:
                raise ValueError(f"in-memory valid mask is invalid: {source}")
            if value.device != mask.device:
                raise ValueError("in-memory module input and mask devices do not match")
            yield value, mask

    return _evaluate_local_output_nmse_plans(
        model,
        plans,
        available_module_paths=available_paths,
        source_records=source_records,
        preprocessing="in-memory-inputs",
        iter_batches=batches,
        decomposition_backends=decomposition_backends,
        execution_backends=execution_backends,
        decomposition_dtype=decomposition_dtype,
        on_plan_result=on_plan_result,
    )
