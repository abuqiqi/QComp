"""执行任务驱动的流式 NMSE：候选常驻 GPU，每批 baseline 只前向一次。

配置指定任务与历史结果；库负责请求编码和局部测量，脚本保存统计断点和报告。
主要内容：
- ``build_local_output_plans``：按实际参数保留率构造 Qwen3 候选。
- ``build_candidates``：合并历史配置并按完整结构去重。
- ``main``：编排候选准备、任务流、断点恢复和报告。

在项目根目录调用：
    python scripts/run_qwen3_local_output_nmse.py --config config/local_output_nmse.json
    python scripts/run_qwen3_local_output_nmse.py --resume artifacts/evaluations/<run>
    python scripts/run_qwen3_local_output_nmse.py --plot-only artifacts/evaluations/<run>/local_output_nmse.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from qcomp import (CompressionExecutionConfig, CompressionPlan, CompressionTarget,
                   MPOSpec, ModelLoadConfig, find_linear, list_linears, load_causal_lm,
                   load_runtime_config, log_event)
from qcomp.evaluation.lm_eval import LMEvalConfig
from qcomp.evaluation.nmse_inputs import TaskInputs, canonical_json
from qcomp.workflows.compression_plan_io import compression_plan_from_dict, compression_plan_to_dict
from qcomp.workflows.streaming_nmse import prepare_candidates, measure_batch

if __package__:
    from .qwen3_mpo_config import qwen3_modes
    from .local_output_nmse_plots import plot_results as _plot_results
    from .nmse_reporting import write_results, compare_history
else:
    from qwen3_mpo_config import qwen3_modes
    from local_output_nmse_plots import plot_results as _plot_results
    from nmse_reporting import write_results, compare_history

PROJECT = Path(__file__).resolve().parents[1]
MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
LINEAR_PATH_RE = re.compile(r'.*\.layers\.(?P<block>\d+)\.(?:self_attn|mlp)\.(?P<module>\w+)$')
DTYPES = {'bfloat16': torch.bfloat16, 'float32': torch.float32}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析任务配置、恢复和补图参数；拒绝旧缓存执行参数。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config/local_output_nmse.json')
    parser.add_argument('--runtime-config', default=str(PROJECT / 'config/runtime-qwen3-8b.toml'))
    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dtype', choices=DTYPES, default='bfloat16')
    parser.add_argument('--decomposition-dtype', choices=DTYPES, default='float32')
    parser.add_argument('--decomposition-provider', default='tensorly')
    parser.add_argument('--execution-provider', default='tensorly')
    parser.add_argument('--decomposition-seed', type=int, default=42)
    parser.add_argument('--start-block', type=int, default=0)
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--plot-only', type=Path)
    parser.add_argument('--heatmap-max', type=float)
    parser.add_argument('--trust-remote-code', action='store_true')
    values = list(argv) if argv is not None else __import__('sys').argv[1:]
    if any(v.split('=')[0] in ('--baseline-cache-mode', '--baseline-inputs', '--rebuild-baseline-inputs') for v in values):
        parser.error('旧缓存执行参数已移除；使用任务配置运行流式 NMSE，旧缓存库接口仍保留')
    args = parser.parse_args(values)
    if args.start_block < 0 or (args.max_blocks is not None and args.max_blocks <= 0):
        parser.error('invalid block range')
    if args.heatmap_max is not None and (not math.isfinite(args.heatmap_max) or args.heatmap_max <= 0):
        parser.error('heatmap-max must be finite and positive')
    if args.resume and args.output:
        parser.error('resume and output are mutually exclusive')
    return args


def load_local_output_config(path: Path) -> dict[str, Any]:
    """读取任务驱动配置，规范历史路径并通过 LMEvalConfig 验证执行参数。"""
    data = json.loads(path.read_text())
    if set(data) != {'retention_ratios', 'tasks'}:
        raise ValueError('配置必须包含 retention_ratios 和 tasks；旧 calibration 配置请迁移')
    ratios = data['retention_ratios']
    if not isinstance(ratios, list) or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError('invalid or duplicate retention ratios')
    names = set()
    for task in data['tasks']:
        if set(task) != {'name', 'evaluation', 'input_strategy', 'historical_results'}:
            raise ValueError('invalid task fields')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', task['name']) or task['name'] in names:
            raise ValueError('invalid or duplicate task name')
        names.add(task['name'])
        c = LMEvalConfig(**task['evaluation'])
        task['evaluation'] = asdict(c)
        if task['input_strategy'] not in ('likelihood', 'reference_answer'):
            raise ValueError('invalid input strategy')
        task['historical_results'] = str((path.parent / task['historical_results']).resolve())
    if not names:
        raise ValueError('tasks must not be empty')
    return data


def file_digest(path: Path) -> str:
    """流式计算文件 SHA-256，不将模型权重整份读入内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """原子替换单个 JSON，保证恢复只读取完整批次统计。"""
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def plot_results(path: Path, vmax: float | None = None) -> list[Path]:
    """从已有 JSON 补绘 NMSE，参数为路径与共享色标上限。"""
    return _plot_results(path, MODULES, vmax)

def _coordinates(path: str) -> tuple[int, str] | None:
    """解析受支持 Qwen3 投影路径；其他 Linear 返回 None。

    参数：
        path: 完整模型模块路径。

    返回：
        ``(block, projection)``；不属于七类投影时返回 None。
    """
    match = LINEAR_PATH_RE.fullmatch(path)
    if match is None or match["module"] not in MODULES:
        return None
    return int(match["block"]), match["module"]


def select_qwen3_targets(
    model: nn.Module,
    *,
    start_block: int,
    max_blocks: int | None,
) -> tuple[str, ...]:
    """选择连续 Qwen3 Block 的七类投影路径。

    参数：
        model: 已加载 Qwen3 模型。
        start_block: 第一个待测 block 编号。
        max_blocks: 最多选择的 block 数；None 表示直到末层。

    返回：
        按 block 和固定投影顺序排列的模块路径。
    """
    indexed = {}
    for path, _ in list_linears(model):
        coordinates = _coordinates(path)
        if coordinates is not None:
            indexed[coordinates] = path
    blocks = sorted({block for block, _ in indexed if block >= start_block})
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    if not blocks or blocks[0] != start_block:
        raise ValueError(f"Qwen3 block {start_block} does not exist")
    paths = []
    for block in blocks:
        missing = [module for module in MODULES if (block, module) not in indexed]
        if missing:
            raise ValueError(
                f"Qwen3 block {block} is missing projections: {', '.join(missing)}"
            )
        paths.extend(indexed[(block, module)] for module in MODULES)
    return tuple(paths)


def group_qwen3_targets_by_block(
    module_paths: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """将有序 Qwen3 投影路径拆为逐 block 的七投影分组。

    参数：
        module_paths: ``select_qwen3_targets`` 返回的投影路径。

    返回：
        保持 block 与投影顺序的路径分组，每组恰好包含七个投影。
    """
    grouped: dict[int, list[str]] = {}
    for path in module_paths:
        coordinates = _coordinates(path)
        if coordinates is None:
            raise ValueError(f"unsupported Qwen3 projection path: {path}")
        grouped.setdefault(coordinates[0], []).append(path)
    result = []
    for block, paths in grouped.items():
        projections = tuple((_coordinates(path) or (-1, ""))[1] for path in paths)
        if projections != MODULES:
            raise ValueError(
                f"Qwen3 block {block} does not contain ordered projections"
            )
        result.append(tuple(paths))
    if not result:
        raise ValueError("module_paths must not be empty")
    return tuple(result)


def _closest_mpo_spec(linear: nn.Linear, target_ratio: float) -> MPOSpec:
    """选择实际参数保留率最接近目标值的统一内部 rank MPO。

    参数：
        linear: 用于确定输入输出 modes 的 Qwen3 Linear。
        target_ratio: 目标 MPO 参数量与稠密权重参数量之比。

    返回：
        合法候选中实际参数保留率最接近目标值的 MPOSpec。
    """
    out_modes = qwen3_modes(linear.out_features)
    in_modes = qwen3_modes(linear.in_features)
    maximum = min(MPOSpec.full_rank(out_modes, in_modes).maximum_internal_ranks)
    candidates = (
        MPOSpec(out_modes, in_modes, (1, rank, rank, 1))
        for rank in range(1, maximum + 1)
    )
    return min(
        candidates,
        key=lambda spec: (
            abs(spec.num_parameters / spec.dense_num_parameters - target_ratio),
            spec.num_parameters,
        ),
    )


def build_local_output_plans(
    model: nn.Module,
    module_paths: Sequence[str],
    retention_ratios: Sequence[float],
) -> dict[str, CompressionPlan]:
    """把目标参数保留率映射成单投影 MPO 压缩计划。

    参数：
        model: 包含全部目标 Linear 的 Qwen3 模型。
        module_paths: 按 block 和投影顺序排列的目标路径。
        retention_ratios: 严格位于 0 与 1 之间的目标参数保留率。

    返回：
        每项仅包含一个 ``CompressionTarget`` 的具名计划；同模块重复 rank 会去重。
    """
    plans = {}
    for path in module_paths:
        linear = find_linear(model, path)
        block, module = _coordinates(path) or (-1, "")
        if block < 0:
            raise ValueError(f"unsupported Qwen3 projection path: {path}")
        used_ranks = set()
        for requested in retention_ratios:
            ratio = float(requested)
            if not math.isfinite(ratio) or not 0 < ratio < 1:
                raise ValueError("retention ratios must be finite and between 0 and 1")
            spec = _closest_mpo_spec(linear, ratio)
            rank = spec.ranks[1]
            if rank in used_ranks:
                continue
            used_ranks.add(rank)
            name = f"block-{block:02d}-{module}-rho-{ratio:.6f}-rank-{rank}"
            plans[name] = CompressionPlan((CompressionTarget(path, "mpo", spec),))
    if not plans:
        raise ValueError("retention ratios produced no compression plans")
    return plans



def build_candidates(model: nn.Module, paths: Sequence[str], config: dict) -> tuple[dict, dict, dict]:
    """将比例候选与历史完整配置合并，验证历史模型和实际矩阵尺寸。"""
    plans, records, index, histories = {}, {}, {}, {}

    def add(plan: CompressionPlan, label: str, ratio: float | None) -> None:
        """按完整结构唯一登记候选，并保留所有来源标签。"""
        target = plan.targets[0]
        key = canonical_json(compression_plan_to_dict(plan))
        if key in index:
            records[index[key]]['labels'].append(label)
            return
        spec = target.spec
        linear = find_linear(model, target.module_path)
        if (linear.out_features, linear.in_features) != (spec.out_features, spec.in_features):
            raise ValueError('historical candidate shape mismatch')
        name = f'candidate-{len(plans):04d}'
        index[key], plans[name] = name, plan
        block, projection = _coordinates(target.module_path)
        records[name] = dict(name=name, module_path=target.module_path, block=block,
            projection=projection, representation=target.representation,
            out_modes=list(spec.out_modes), in_modes=list(spec.in_modes), ranks=list(spec.ranks),
            rank=max(spec.ranks), requested_retention_ratio=ratio, labels=[label],
            actual_retention_ratio=spec.num_parameters / spec.dense_num_parameters,
            dense_parameters=spec.dense_num_parameters, compressed_parameters=spec.num_parameters,
            healing=False, compression_plan=compression_plan_to_dict(plan))

    for path in paths:
        for ratio in config['retention_ratios']:
            plan = CompressionPlan((CompressionTarget(path, 'mpo', _closest_mpo_spec(find_linear(model, path), ratio)),))
            add(plan, f'rho:{ratio:g}', ratio)
    for task in config['tasks']:
        history = json.loads(Path(task['historical_results']).read_text())
        hc = history['model']['config']
        for field in ('model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim'):
            if hc.get(field) != model.config.to_dict().get(field):
                raise ValueError(f'historical model mismatch: {field}')
        histories[task['name']] = history
        seen = set()
        for case in history['cases']:
            plan = compression_plan_from_dict(case['compression_plan'])
            if len(plan.targets) != 1:
                raise ValueError('historical case must have one target')
            path = plan.targets[0].module_path
            if path in paths:
                if path in seen:
                    raise ValueError('duplicate historical module')
                seen.add(path)
                add(plan, 'history:' + task['name'], None)
        if seen != set(paths):
            raise ValueError('history is missing selected modules')
    return plans, records, histories


def validate_resume(state: dict, identity: dict) -> None:
    """恢复前校验协议和实验身份，拒绝混用配置、权重或软件版本。"""
    if state.get('format_version') != 1 or canonical_json(state['identity']) != canonical_json(identity):
        raise ValueError('resume experiment identity mismatch')


def main(argv: Sequence[str] | None = None) -> None:
    """执行准备、流式测量、原子统计断点和最终绘图，不保存激活。"""
    args = parse_args(argv)
    if args.plot_only:
        plot_results(args.plot_only, args.heatmap_max)
        return
    started = time.perf_counter()
    prior = None
    if args.resume:
        prior = json.loads((args.resume / 'checkpoint.json').read_text())
        config = prior['identity']['config']
        saved_args = prior['identity']['arguments']
        for key, value in saved_args.items():
            setattr(args, key, value)
        output = args.resume
    else:
        config = load_local_output_config(args.config)
        output = args.output or PROJECT / 'artifacts/evaluations/qwen3-local-output-nmse' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime_config(args.runtime_config)
    model_source = Path(args.model or runtime.model_name_or_path).resolve()
    files = sorted(set(model_source.glob('*.safetensors')) | set(model_source.glob('*.json')))
    if not files or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('local checkpoint safetensors required for weight identity')
    print('Hashing checkpoint identity', flush=True)
    file_identity = {p.name: file_digest(p) for p in files}
    resources = load_causal_lm(ModelLoadConfig(model_name_or_path=str(model_source), device=args.device,
        dtype=DTYPES[args.model_dtype], trust_remote_code=args.trust_remote_code), runtime_config_path=args.runtime_config)
    model, tokenizer = resources.model, resources.tokenizer
    paths = select_qwen3_targets(model, start_block=args.start_block, max_blocks=args.max_blocks)
    plans, records, histories = build_candidates(model, paths, config)
    identity = {'config': config, 'files': file_identity,
        'arguments': {key: getattr(args, key) for key in ('model', 'device', 'runtime_config', 'model_dtype', 'decomposition_dtype', 'decomposition_provider', 'execution_provider', 'decomposition_seed', 'start_block', 'max_blocks', 'trust_remote_code')},
        'plans': {name: compression_plan_to_dict(plan) for name, plan in plans.items()},
        'implementation': {str(p.relative_to(PROJECT)): file_digest(p) for p in [Path(__file__).resolve(), PROJECT/'src/qcomp/evaluation/nmse_inputs.py', PROJECT/'src/qcomp/workflows/streaming_nmse.py', PROJECT/'src/qcomp/evaluation/local_output_nmse.py']},
        'histories': {t['name']: file_digest(Path(t['historical_results'])) for t in config['tasks']},
        'tokenizer': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    if prior:
        validate_resume(prior, identity)
    for source, history in histories.items():
        for field in ('model_dtype', 'decomposition_dtype', 'decomposition_provider', 'execution_provider'):
            if history['execution'][field].removeprefix('torch.') != getattr(args, field):
                raise ValueError(f'historical execution mismatch: {source}/{field}')
    state = prior or {'format_version': 1, 'identity': identity, 'tasks': {}, 'timing': {'prior_seconds': 0.0}}
    experiment = dict(model=str(model_source), model_dtype=args.model_dtype,
        decomposition_dtype=args.decomposition_dtype, decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider, retention_ratios=config['retention_ratios'],
        calibration={'datasets': [{'name': t['name']} for t in config['tasks']]},
        protocol='streaming-task-inputs-v1', healing=False, run_id=output.name,
        historical_identity_note='历史结果未保存实际 token 和完整权重摘要；配置对齐不证明逐 token 历史复现。')
    decomposition, execution = CompressionExecutionConfig(decomposition_provider=args.decomposition_provider,
        execution_provider=args.execution_provider, decomposition_dtype=DTYPES[args.decomposition_dtype]).build_backends(plans)
    print(f'Preparing {len(plans)} candidates', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    candidates, seconds = prepare_candidates(model, plans, decomposition, execution,
                                             DTYPES[args.decomposition_dtype], args.decomposition_seed)
    experiment['candidate_tensor_bytes'] = sum(p.numel()*p.element_size() for module in candidates.values() for p in module.parameters())
    experiment['decomposition_seconds'] = sum(seconds.values())
    for name in records:
        records[name]['compression_seconds'] = seconds[name]
    block_paths = {path: path.rsplit('.', 2)[0] for path in paths}
    atomic_json(output / 'experiment_config.json', {**experiment, 'identity': identity})
    try:
        for entry in config['tasks']:
            source = entry['name']
            prep = time.perf_counter()
            inputs = TaskInputs(model, tokenizer, LMEvalConfig(**entry['evaluation']), entry['input_strategy'], args.runtime_config)
            task_identity = json.loads(canonical_json(inputs.identity))
            if source in state['tasks']:
                if state['tasks'][source]['identity'] != task_identity:
                    raise ValueError('resume task/data/version identity mismatch')
            else:
                state['tasks'][source] = {'identity': task_identity, 'batches': 0, 'digest': '0'*64,
                    'examples': 0, 'logical_requests': 0, 'sequences': 0, 'valid_tokens': 0,
                    'sums': {name: [0.0, 0.0] for name in plans}, 'complete': False,
                    'evaluation_seconds': 0.0, 'preparation_seconds': 0.0, 'write_seconds': 0.0}
            saved = state['tasks'][source]
            saved['preparation_seconds'] += time.perf_counter() - prep
            skip = saved['batches']
            observed = 0
            iterator = iter(inputs.batches())
            while True:
                prep = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                saved['preparation_seconds'] += time.perf_counter() - prep
                observed += 1
                if observed <= skip:
                    if observed == skip and batch['digest'] != saved['digest']:
                        raise ValueError('resume input prefix digest mismatch')
                    continue
                if saved['complete']:
                    raise ValueError('completed task has additional input')
                tick = time.perf_counter()
                delta = measure_batch(model, model.base_model, plans, candidates, block_paths, batch)
                elapsed = time.perf_counter() - tick
                # 只在完整 baseline 和全部候选成功后更新统计。
                updated = copy.deepcopy(saved)
                for name, row in delta.items():
                    updated['sums'][name] = [a+b for a,b in zip(updated['sums'][name], row)]
                for field in ('examples', 'logical_requests', 'sequences'):
                    updated[field] += batch[field]
                updated['valid_tokens'] += int(batch['attention_mask'].sum())
                updated['batches'] = observed
                updated['digest'] = batch['digest']
                updated['evaluation_seconds'] += elapsed
                state['tasks'][source] = updated
                write_start = time.perf_counter()
                atomic_json(output / 'checkpoint.json', state)
                updated['write_seconds'] += time.perf_counter() - write_start
                saved = updated
                if observed % 25 == 0 or observed == 1:
                    print(f'{source}: batches={observed} examples={saved["examples"]} tokens={saved["valid_tokens"]} seconds={elapsed:.3f}', flush=True)
                    log_event(output/'events.jsonl', 'batch_progress', task=source, batches=observed, examples=saved['examples'])
            if observed < skip or not observed:
                raise ValueError('input stream shorter than checkpoint or empty')
            c = inputs.config
            expected = sum(
                min(len(task.eval_docs)-c.sample_start_index,
                    math.ceil(len(task.eval_docs)*c.limit) if isinstance(c.limit, float) else c.limit)
                if c.limit is not None else len(task.eval_docs)-c.sample_start_index
                for task in inputs.tasks.values())
            if saved['examples'] != expected:
                raise ValueError('evaluated document count differs from task selection')
            saved['complete'] = True
            atomic_json(output/'checkpoint.json', state)
            write_results(output, experiment, records, state, paths)
            del inputs, iterator
    finally:
        experiment['total_seconds'] = time.perf_counter() - started + state['timing']['prior_seconds']
        state['timing']['prior_seconds'] = experiment['total_seconds']
        if args.device.startswith('cuda'):
            experiment['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
            experiment['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
        # 异常发生于尚未提交批次时，内存统计仍指向最后完整批次。
        atomic_json(output/'checkpoint.json', state)
        write_results(output, experiment, records, state, paths)
    plot_started = time.perf_counter()
    plot_results(output/'local_output_nmse.json', args.heatmap_max)
    compare_history(output, histories, MODULES)
    plotting_seconds = time.perf_counter() - plot_started
    result_path = output/'local_output_nmse.json'
    result = json.loads(result_path.read_text())
    result['experiment']['plotting_seconds'] = plotting_seconds
    result['experiment']['total_seconds'] += plotting_seconds
    state['timing']['prior_seconds'] += plotting_seconds
    atomic_json(result_path, result)
    atomic_json(output/'experiment_config.json', {**result['experiment'], 'identity': identity})
    atomic_json(output/'checkpoint.json', state)
    with (output/'report.md').open('a') as handle:
        handle.write(f'\n图表与对比耗时：{plotting_seconds:.2f} 秒；总耗时：{state["timing"]["prior_seconds"]:.2f} 秒。\n')
    log_event(output/'events.jsonl', 'experiment_completed', candidates=len(plans))
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
