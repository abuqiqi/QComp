"""执行候选常驻的流式局部 NMSE，不替换 baseline 或缓存跨批激活。

候选只构造一次；每个 batch 的 backbone 正常前向触发 block hook，提交前仅返回
临时统计，持久化与恢复进度由上层负责。
主要内容：
- ``prepare_candidates``：分解并构造独立运行模块。
- ``measure_batch``：一次 baseline 前向测量全部候选。
"""
from __future__ import annotations

import time
import math
from collections import defaultdict
from typing import Any, Mapping

import torch
from torch import nn

from ..model import find_linear
from ..evaluation.local_output_nmse import local_output_error_sums
from .compress import CompressionPlan


def prepare_candidates(model: nn.Module, plans: Mapping[str, CompressionPlan],
                       decomposition: Any, execution: Any,
                       dtype: torch.dtype, seed: int) -> tuple[dict, dict]:
    """为 plans 中每个单目标分解一次，返回独立模块和逐候选耗时。"""
    modules, seconds = {}, {}
    for name, plan in plans.items():
        if len(plan.targets) != 1:
            raise ValueError('streaming NMSE requires single-target plans')
        target = plan.targets[0]
        original = find_linear(model, target.module_path)
        torch.manual_seed(seed)
        if original.weight.is_cuda:
            torch.cuda.synchronize(original.weight.device)
        started = time.perf_counter()
        with torch.no_grad():
            artifact = decomposition[target.representation].decompose(
                original.weight.detach().to(dtype=dtype), target.spec)
            artifact = artifact.to(device=original.weight.device, dtype=original.weight.dtype)
            modules[name] = execution[target.representation].build_linear(artifact, trainable=False).eval()
            del artifact
        if original.weight.is_cuda:
            torch.cuda.synchronize(original.weight.device)
        seconds[name] = time.perf_counter() - started
    return modules, seconds


def measure_batch(model: nn.Module, backbone: nn.Module, plans: Mapping[str, CompressionPlan],
                  candidates: Mapping[str, nn.Module], block_paths: Mapping[str, str],
                  batch: Mapping[str, Any]) -> dict[str, list[float]]:
    """运行一批并返回每个候选的误差、能量和；异常不返回部分统计且总是移除 hooks。"""
    groups = defaultdict(list)
    by_path = defaultdict(list)
    for name, plan in plans.items():
        path = plan.targets[0].module_path
        by_path[path].append(name)
    for path in by_path:
        groups[block_paths[path]].append(path)
    captured, sums, handles = {}, {}, []
    device = next(model.parameters()).device
    mask = batch['attention_mask'].to(device)
    valid_tokens = int(batch['attention_mask'].sum())
    training = [(module, module.training) for module in model.modules()]

    def capture(path: str):
        """创建投影 hook，保存本 block 的真实输入输出。"""
        def hook(module, args, output):
            """只保留引用，不改变原模型数据流。"""
            if path in captured:
                raise ValueError(f'projection executed twice: {path}')
            captured[path] = (args[0], output)
        return hook

    def evaluate(paths: list[str]):
        """创建 block hook，在输出传播到后续 block 前测量全部局部候选。"""
        def hook(module, args, output):
            """按模块执行候选，统计保持在 GPU，整批结束才搬回 CPU。"""
            for path in paths:
                value, reference = captured.pop(path)
                for name in by_path[path]:
                    if name in sums:
                        raise ValueError(f'candidate executed twice: {name}')
                    compressed = candidates[name](value)
                    error, power, _ = local_output_error_sums(reference, compressed, mask, valid_tokens=valid_tokens)
                    sums[name] = torch.stack((error, power))
                    del compressed
        return hook

    try:
        model.eval()
        for path in by_path:
            handles.append(find_linear(model, path).register_forward_hook(capture(path)))
        for path, paths in groups.items():
            handles.append(model.get_submodule(path).register_forward_hook(evaluate(paths)))
        with torch.inference_mode():
            output = backbone(input_ids=batch['input_ids'].to(device), attention_mask=mask,
                              use_cache=False)
            del output
        if captured or set(sums) != set(plans):
            raise ValueError('not all candidate modules were executed')
        names = list(sums)
        values = torch.stack([sums[name] for name in names]).cpu().tolist()
        if not all(all(math.isfinite(v) and v >= 0 for v in row) for row in values):
            raise ValueError('non-finite NMSE sums')
        return dict(zip(names, values))
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        for module, state in training:
            module.training = state
