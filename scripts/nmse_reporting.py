"""把流式 NMSE 统计输出为旧格式兼容明细，并比较同配置的历史任务指标。

本模块只处理已提交统计，不参与模型执行；图片和表格保存于实验目录。
主要内容：
- ``write_results``：导出完整或部分统计及报告。
- ``compare_history``：精确匹配 MPO 配置，生成对照图、相关性和分歧表。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from qcomp.evaluation.local_output_nmse import local_output_nmse
from qcomp.evaluation.nmse_inputs import canonical_json


def _write(path: Path, value: Any) -> None:
    """原子写入统计 JSON，避免进程中断产生半个文件。"""
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def write_results(output: Path, experiment: dict, records: dict, state: dict, paths: Sequence[str]) -> None:
    """由已提交分子分母生成候选、逐来源结果和进度报告。"""
    cases = []
    tasks = state['tasks']
    complete = len(tasks) == len(state['identity']['config']['tasks']) and all(t['complete'] for t in tasks.values())
    for name, record in records.items():
        sources = {}
        for source, task in tasks.items():
            error, power = task['sums'][name]
            if task['valid_tokens']:
                sources[source] = dict(output_nmse=local_output_nmse(error, power),
                    squared_error_sum=error, target_power_sum=power,
                    valid_tokens=task['valid_tokens'], evaluated_examples=task['examples'],
                    sequences=task['sequences'], logical_requests=task['logical_requests'])
        error = sum(r['squared_error_sum'] for r in sources.values())
        power = sum(r['target_power_sum'] for r in sources.values())
        cases.append({**record, 'status': 'ok' if complete else 'partial', 'sources': sources,
                      'output_nmse': local_output_nmse(error, power) if sources else None,
                      'squared_error_sum': error, 'target_power_sum': power,
                      'valid_tokens': sum(r['valid_tokens'] for r in sources.values()),
                      'historical_metrics': None})
    dense = []
    for path in paths:
        record = next(r for r in records.values() if r['module_path'] == path)
        dense.append({k: record[k] for k in ('module_path', 'block', 'projection', 'dense_parameters')})
        dense[-1].update(status='dense', output_nmse=0.0, actual_retention_ratio=1.0,
                         compressed_parameters=record['dense_parameters'])
    experiment = {**experiment, 'complete': complete,
                  'task_progress': {name: {k:v for k,v in task.items() if k != 'sums'} for name,task in tasks.items()}}
    _write(output/'local_output_nmse.json', dict(experiment=experiment, dense=dense, cases=cases))
    _write(output/'experiment_config.json', {**experiment, 'identity': state['identity']})
    fields = ['name','module_path','status','rank','requested_retention_ratio','actual_retention_ratio',
              'output_nmse','valid_tokens','dense_parameters','compressed_parameters','compression_seconds']
    with (output/'local_output_nmse.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(cases)
    lines = ['# Qwen3 流式局部输出 NMSE', '',
             f'- 状态：{"完成" if complete else "部分结果"}；候选 {len(cases)}，投影 {len(paths)}。',
             '- 全部有效输入位置计入 NMSE；padding 排除；同题相同模型输入去重。',
             '- 每批 baseline 前向一次；无激活磁盘缓存；候选常驻 GPU。',
             '- 总体按误差平方和与参考能量累加后计算，不平均来源 NMSE。',
             '- ' + experiment['historical_identity_note'], '',
             '| 任务 | 已处理题数 | 逻辑请求 | 输入序列 | 有效 tokens | 状态 |',
             '|---|---:|---:|---:|---:|---|']
    for name, task in tasks.items():
        lines.append(f'| {name} | {task["examples"]} | {task["logical_requests"]} | {task["sequences"]} | {task["valid_tokens"]} | {"完成" if task["complete"] else "部分"} |')
    lines += ['', '## 资源与耗时', '',
              f'- 候选核：{experiment["candidate_tensor_bytes"]/2**30:.3f} GiB。',
              f'- 本次候选分解：{experiment["decomposition_seconds"]:.2f} 秒。',
              f'- 输入准备累计：{sum(t["preparation_seconds"] for t in tasks.values()):.2f} 秒。',
              f'- baseline 与局部评测累计：{sum(t["evaluation_seconds"] for t in tasks.values()):.2f} 秒。',
              f'- 统计写盘累计：{sum(t["write_seconds"] for t in tasks.values()):.2f} 秒。',
              f'- 含恢复的总耗时：{experiment.get("total_seconds", 0):.2f} 秒。',
              f'- 本次 GPU allocated 峰值：{experiment.get("peak_gpu_allocated_bytes", 0)/2**30:.3f} GiB。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def average_ranks(values: list[float]) -> list[float]:
    """计算升序平均秩，并列值使用占据位置的平均数。"""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0]*len(values)
    start = 0
    while start < len(ordered):
        end = start+1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start+1+end)/2
        start = end
    return ranks


def compare_history(output: Path, histories: dict, modules: Sequence[str]) -> None:
    """按完整压缩计划匹配历史指标，输出 Spearman、分歧表和对照图。"""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.patches import Patch
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    document = json.loads((output/'local_output_nmse.json').read_text())
    lookup = {canonical_json(c['compression_plan']): c for c in document['cases']}
    section = ['\n## 历史任务指标对比\n']
    for source, history in histories.items():
        rows = []
        for old in history['cases']:
            case = lookup.get(canonical_json(old['compression_plan']))
            if case is None or source not in case['sources']:
                continue
            metrics = {}
            for metric, direction in history['evaluation_config']['metric_directions'].items():
                before, after = history['baseline']['metrics'][metric], old['metrics'][metric]
                metrics[metric] = {'baseline': before, 'compressed': after,
                                  'drop_pp': 100*(before-after)*(1 if direction == 'higher' else -1)}
            case['historical_metrics'] = case['historical_metrics'] or {}
            case['historical_metrics'][source] = metrics
            rows.append({'module_path': case['module_path'], 'block': case['block'], 'projection': case['projection'],
                         'output_nmse': case['sources'][source]['output_nmse'],
                         'saved_parameters': case['dense_parameters']-case['compressed_parameters'],
                         'metrics': metrics})
        prefix = f'qwen3-nmse-{source}-{output.name}'
        current = document['experiment'].get('task_progress', {}).get(source, {}).get('identity', {}).get('evaluation')
        historical = history['evaluation_config'].get('evaluation')
        comparison_differences = ({key: {'current': current.get(key), 'historical': historical.get(key)}
            for key in ('task', 'num_fewshot', 'max_length', 'limit', 'evaluation_seed', 'apply_chat_template', 'sample_start_index')
            if current.get(key) != historical.get(key)} if current and historical else {})
        result = {'evaluation_differences': comparison_differences, 'rows': rows, 'metrics': {}, 'limitations': document['experiment']['historical_identity_note'],
                  'historical_evaluation': history['evaluation_config'], 'historical_execution': history['execution']}
        if comparison_differences:
            section.append(f'输入配置与历史评测存在差异：`{canonical_json(comparison_differences)}`。\n')
        for metric in history['evaluation_config']['metric_directions']:
            x = [r['output_nmse'] for r in rows]
            y = [r['metrics'][metric]['drop_pp'] for r in rows]
            rx, ry = average_ranks(x), average_ranks(y)
            corr = float(np.corrcoef(rx, ry)[0,1]) if len(rows)>1 and len(set(x))>1 and len(set(y))>1 else None
            divergent = sorted(range(len(rows)), key=lambda i: (-abs(rx[i]-ry[i]), rows[i]['module_path']))[:20]
            result['metrics'][metric] = {'spearman': corr, 'largest_rank_disagreements': [
                {**rows[i], 'nmse_rank': rx[i], 'drop_rank': ry[i], 'rank_difference': abs(rx[i]-ry[i])} for i in divergent]}
            fig = Figure(figsize=(12, 9), layout='constrained'); FigureCanvasAgg(fig)
            axes = fig.subplots(3, 1)
            blocks = sorted({r['block'] for r in rows})
            for ax, key, label in ((axes[0], 'output_nmse', 'Output NMSE'), (axes[1], metric, 'Accuracy drop (pp)')):
                matrix = np.full((len(modules), len(blocks)), np.nan)
                for r in rows:
                    matrix[modules.index(r['projection']), blocks.index(r['block'])] = r['output_nmse'] if key == 'output_nmse' else r['metrics'][key]['drop_pp']
                if key == 'output_nmse':
                    image = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap='viridis')
                else:
                    # 与历史敏感度图保持一致，突出 0～10 个百分点内的差异。
                    cmap = colormaps['viridis'].with_extremes(bad='#dddddd', under='#24243e')
                    image = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap, vmin=0, vmax=10)
                    for row, col in np.argwhere(np.isfinite(matrix) & ((matrix > 10) | (matrix < 0))):
                        value = matrix[row, col]
                        ax.text(col, row, f'{value:.1g}' if value < 0 else f'{value:.1f}',
                                color='white' if value < 0 else 'black', ha='center', va='center',
                                fontsize=6 if value < 0 else 7)
                    legend = []
                    if np.isnan(matrix).any():
                        legend.append(Patch(color='#dddddd', label='Not evaluated'))
                    if (matrix < 0).any():
                        legend.append(Patch(color='#24243e', label='Improvement (< 0)'))
                    if legend:
                        ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)

                ax.set_xticks(range(len(blocks)), blocks); ax.set_yticks(range(len(modules)), modules)
                ax.set_title(f'{source} historical configuration: {label}')
                fig.colorbar(image, ax=ax, label=label if key == 'output_nmse' else f'{label}, clipped to [0, 10]')
            axes[2].scatter(x, y, s=12)
            axes[2].set_xlabel('Output NMSE'); axes[2].set_ylabel('Accuracy drop (pp)')
            axes[2].set_title(f'Spearman: {corr:.4f}' if corr is not None else 'Spearman: undefined (constant or insufficient values)')
            filename = f'{prefix}-{metric}-comparison.png'
            fig.savefig(output/filename, dpi=180); fig.clear()
            section += [f'### {source} / {metric}\n', f'![对比]({filename})\n',
                        '| 模块 | NMSE 排名 | 掉点排名 | 排名差 |', '|---|---:|---:|---:|']
            section += [f'| {rows[i]["module_path"]} | {rx[i]:g} | {ry[i]:g} | {abs(rx[i]-ry[i]):g} |' for i in divergent]
            with (output/f'{prefix}-{metric}-comparison.csv').open('w') as handle:
                writer = csv.writer(handle); writer.writerow(['module_path','nmse','baseline','compressed','drop_pp','saved_parameters'])
                for r in rows:
                    m = r['metrics'][metric]
                    writer.writerow([r['module_path'],r['output_nmse'],m['baseline'],m['compressed'],m['drop_pp'],r['saved_parameters']])
        _write(output/f'{prefix}-comparison.json', result)
    _write(output/'local_output_nmse.json', document)
    with (output/'report.md').open('a') as handle:
        handle.write('\n'.join(section)+'\n')
