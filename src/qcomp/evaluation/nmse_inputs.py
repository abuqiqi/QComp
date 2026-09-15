"""把 lm-eval 任务逐题转换为局部 NMSE 的固定输入，不执行生成或保存请求缓存。

任务模板与 few-shot 采样由 lm-eval 提供，编码使用 HFLM 的上下文/续写规则；
workflow 接收逐批 token、mask、计数和顺序摘要。
主要内容：
- ``TaskInputs``：加载任务、记录身份并流式构造评测输入。
- ``encode_input``：复用 HFLM 编码和因果语言模型的输入移位。
- ``canonical_json``：生成稳定身份和进度摘要。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from importlib.metadata import version
from typing import Any, Iterator

import torch

from .lm_eval import LMEvalConfig
from ..runtime import configure_runtime


def canonical_json(value: Any) -> str:
    """将配置或输入身份转换为稳定 JSON；函数按模块和名称标识。"""
    def encode(obj: Any) -> str:
        """序列化任务定义中的函数，不记录进程内存地址。"""
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        raise TypeError(f"unsupported identity value: {type(obj)}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=encode)


def encode_input(encoder: Any, context: str, continuation: str, max_length: int) -> list[int]:
    """按 HFLM 规则编码固定上下文和续写，左截断并移除最后一个目标 token。"""
    if context:
        left, right = encoder._encode_pair(context, continuation)
    else:
        left, right = [encoder.prefix_token_id], encoder.tok_encode(continuation)
    if not right or len(right) > max_length:
        raise ValueError("continuation must contain 1..max_length tokens")
    result = (left + right)[-(max_length + 1):][:-1]
    if not result:
        raise ValueError("empty causal input")
    return result


class TaskInputs:
    """持有一个任务或任务组，按配置逐题构造有界 token batch。"""

    def __init__(self, model: Any, tokenizer: Any, config: LMEvalConfig,
                 strategy: str, runtime_config_path: str | None = None) -> None:
        """加载 config 对应任务；strategy 为 likelihood 或 reference_answer。"""
        configure_runtime(runtime_config_path)
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        import numpy as np

        self.config, self.strategy = config, strategy
        if strategy not in ("likelihood", "reference_answer"):
            raise ValueError("unsupported NMSE input strategy")
        random.seed(config.evaluation_seed)
        np.random.seed(config.evaluation_seed)
        torch.manual_seed(config.evaluation_seed)
        loaded = TaskManager().load([config.task])
        self.tasks = dict(sorted(loaded['tasks'].items()))
        self.encoder = HFLM(pretrained=model, tokenizer=tokenizer,
                            batch_size=config.batch_size, max_length=config.max_length,
                            device=str(next(model.parameters()).device))
        self.tokenizer = tokenizer
        self.identity = {'evaluation': asdict(config), 'strategy': strategy,
                         'max_padded_tokens': config.max_length,
                         'versions': {p: version(p) for p in ('lm_eval', 'transformers', 'datasets', 'torch')},
                         'tasks': {}}
        if config.sample_start_index and len(self.tasks) != 1:
            raise ValueError('sample_start_index requires a single leaf task')
        for name, task in self.tasks.items():
            default = task.get_config('num_fewshot')
            task.set_config(key='num_fewshot', value=(
                default if default == 0 else config.num_fewshot
                if config.num_fewshot is not None else default or 0))
            task.set_fewshot_seed(seed=config.evaluation_seed)
            output_type = task.get_config('output_type')
            if strategy == 'likelihood' and output_type not in ('multiple_choice', 'loglikelihood'):
                raise ValueError(f'{name}: likelihood requires a scoring task')
            if strategy == 'reference_answer' and name not in ('gsm8k', 'triviaqa'):
                raise ValueError('reference_answer supports gsm8k and triviaqa')
            docs = task.eval_docs
            split_digests = {}
            for split, dataset in task.dataset.items():
                digest = hashlib.sha256()
                for row in dataset:
                    digest.update((canonical_json(row) + '\n').encode())
                split_digests[split] = {'count': len(dataset), 'sha256': digest.hexdigest()}
            self.identity['tasks'][name] = {
                'config': task.dump_config(), 'count': len(docs),
                'dataset_fingerprint': getattr(docs, '_fingerprint', None),
                'dataset_splits': split_digests,
            }

    def batches(self) -> Iterator[dict[str, Any]]:
        """逐题编码，同题相同模型输入去重；累计摘要允许恢复时验证已处理前缀。"""
        c = self.config
        pending = []
        digest = '0' * 64
        for name, task in self.tasks.items():
            task.set_fewshot_seed(seed=c.evaluation_seed)
            docs = task.eval_docs
            stop = len(docs)
            if c.limit is not None:
                count = math.ceil(len(docs) * c.limit) if isinstance(c.limit, float) else c.limit
                stop = min(stop, c.sample_start_index + count)
            for doc_id in range(c.sample_start_index, stop):
                doc = docs[doc_id]
                context = task.fewshot_context(
                    doc, num_fewshot=task.get_config('num_fewshot'),
                    apply_chat_template=c.apply_chat_template,
                    chat_template=self.encoder.apply_chat_template if c.apply_chat_template else None,
                    gen_prefix=task.doc_to_prefix(doc))
                requests = task.construct_requests(doc=doc, ctx=context,
                    metadata=(name, doc_id, task.config.repeats),
                    apply_chat_template=c.apply_chat_template,
                    chat_template=self.encoder.apply_chat_template if c.apply_chat_template else None)
                if not isinstance(requests, list):
                    requests = [requests]
                if self.strategy == 'reference_answer':
                    answer = doc['answer'] if name == 'gsm8k' else doc['answer']['value']
                    delimiter = task.get_config('target_delimiter')
                    pairs = [(requests[0].args[0], (delimiter or '') + answer)]
                else:
                    pairs = [request.args[:2] for request in requests]
                unique = {}
                for context, continuation in pairs:
                    ids = encode_input(self.encoder, context, continuation, c.max_length)
                    unique.setdefault(tuple(ids), ids)
                for i, ids in enumerate(unique.values()):
                    identity = [name, doc_id, i, ids]
                    digest = hashlib.sha256((digest + canonical_json(identity)).encode()).hexdigest()
                    if pending and (len(pending)+1)*max(len(ids), max(len(row[0]) for row in pending)) > c.max_length:
                        yield self._batch(pending)
                        pending = []
                    pending.append((ids, int(i == 0), len(pairs) if i == 0 else 0, digest))
                    if len(pending) == c.batch_size:
                        yield self._batch(pending)
                        pending = []
        if pending:
            yield self._batch(pending)

    def _batch(self, rows: list) -> dict[str, Any]:
        """将当前少量序列右侧补齐，并携带不重复的题目和逻辑请求计数。"""
        size = max(len(row[0]) for row in rows)
        ids = torch.full((len(rows), size), self.tokenizer.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row[0])] = torch.tensor(row[0])
            mask[index, :len(row[0])] = 1
        return {'input_ids': ids, 'attention_mask': mask,
                'examples': sum(row[1] for row in rows),
                'logical_requests': sum(row[2] for row in rows),
                'sequences': len(rows), 'digest': rows[-1][3]}
