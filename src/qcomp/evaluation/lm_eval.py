"""通过 lm-eval 统一执行本地缓存中的标准语言模型 benchmark。

本模块把已经加载的模型和 tokenizer 交给 lm-eval，由其内置 task 负责数据读取、prompt、
请求展开、batching、答案过滤和指标聚合。qcomp 只应用 runtime 离线配置、复用 task
对象，并把结果转换为通用 ``EvaluationResult``；训练 DataLoader 不在这里处理。

主要内容：
- ``LMEvalConfig``：定义 task 名称和 lm-eval 执行参数。
- ``LMEvalEvaluator``：复用已加载 task 评测不同模型状态。
- ``lm_eval_dataset_size``：读取 task/group 的完整评测集规模。
- ``_extract_metrics``：提取 task 或 group 的数值指标和样本数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from ..runtime import configure_runtime
from .task import EvaluationResult, EvaluationTask


@dataclass(frozen=True)
class LMEvalConfig:
    """定义单个 lm-eval task 或 group 的执行参数。"""

    task: str
    num_fewshot: int | None = None
    batch_size: int = 8
    max_length: int = 4096
    limit: int | float | None = None
    seed: int = 42
    apply_chat_template: bool = False
    sample_start_index: int = 0

    def __post_init__(self) -> None:
        """规范 task 名称并校验执行参数。

        异常：
            TypeError: chat template 开关不是布尔值时抛出。
            ValueError: task 或数值字段无效时抛出。
        """

        task = self.task.strip()
        if not task:
            raise ValueError("task must not be empty")
        object.__setattr__(self, "task", task)
        if (
            not isinstance(self.sample_start_index, int)
            or isinstance(self.sample_start_index, bool)
            or self.sample_start_index < 0
        ):
            raise ValueError("sample_start_index must be a non-negative integer")
        if self.sample_start_index and isinstance(self.limit, float):
            raise ValueError("sample_start_index requires an integer limit or None")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        if not isinstance(self.apply_chat_template, bool):
            raise TypeError("apply_chat_template must be bool")
        if self.limit is not None:
            valid_limit = (
                isinstance(self.limit, int)
                and not isinstance(self.limit, bool)
                and self.limit > 0
            ) or (
                isinstance(self.limit, float)
                and 0 < self.limit < 1
            )
            if not valid_limit:
                raise ValueError(
                    "limit must be a positive integer, fraction in (0, 1), or None"
                )


def lm_eval_dataset_size(
    task: str, *, runtime_config_path: str | Path | None = None
) -> int:
    """读取 task 对应评测集总条数，group 汇总叶子任务，不加载模型。

    参数：
        task: lm-eval task 或 group 名称。
        runtime_config_path: 与实验相同的运行配置路径。

    返回：
        task 定义选用的 test 或 validation split 的总条数。
    """
    configure_runtime(runtime_config_path)
    from lm_eval.tasks import TaskManager

    loaded = TaskManager().load([task])
    total = sum(len(item.eval_docs) for item in loaded["tasks"].values())
    if total <= 0:
        raise ValueError(f"no evaluation samples found for {task!r}")
    return total


def _metric_name(value: str) -> str | None:
    """把 lm-eval 的 metric/filter 键转换为稳定结果名称。

    参数：
        value: 例如 ``acc,none`` 或 ``exact_match,strict-match`` 的键。

    返回：
        去除 stderr 后的规范名称；非指标元数据返回 ``None``。
    """

    metric, separator, filter_name = value.partition(",")
    if metric.endswith("_stderr") or metric in {
        "alias",
        "name",
        "sample_count",
        "sample_len",
    }:
        return None
    if separator and filter_name != "none":
        return f"{metric}_{filter_name}".replace("-", "_")
    return metric


def _result_entry(result: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    """取得指定 task 或 group 的汇总结果。

    参数：
        result: lm-eval 返回的完整结果。
        task: 请求执行的 task 或 group 名称。

    返回：
        对应 task/group 的汇总映射。

    异常：
        ValueError: 结果中没有请求名称时抛出。
    """

    for section_name in ("groups", "results"):
        section = result.get(section_name)
        if isinstance(section, Mapping):
            entry = section.get(task)
            if isinstance(entry, Mapping):
                return entry
    raise ValueError(f"lm-eval result does not contain task {task!r}")


def _sample_count(
    result: Mapping[str, Any],
    task: str,
    entry: Mapping[str, Any],
) -> int:
    """从 task/group 汇总或 n-samples 中取得实际样本数。

    参数：
        result: lm-eval 完整结果。
        task: 请求执行的 task 或 group 名称。
        entry: 已选择的 task/group 汇总。

    返回：
        正的实际评测样本数。

    异常：
        ValueError: 结果中没有有效样本数时抛出。
    """

    sample_len = entry.get("sample_len")
    if isinstance(sample_len, int) and sample_len > 0:
        return sample_len
    metric_counts = entry.get("sample_count")
    if isinstance(metric_counts, Mapping):
        counts = [
            int(value)
            for value in metric_counts.values()
            if isinstance(value, int) and value > 0
        ]
        if counts:
            return max(counts)
    all_counts = result.get("n-samples")
    if isinstance(all_counts, Mapping):
        task_count = all_counts.get(task)
        if isinstance(task_count, Mapping):
            effective = task_count.get("effective")
            if isinstance(effective, int) and effective > 0:
                return effective
    raise ValueError(f"lm-eval result does not contain sample count for {task!r}")


def _extract_metrics(
    result: Mapping[str, Any],
    task: str,
) -> tuple[dict[str, float], int]:
    """提取一个 lm-eval task/group 的数值指标与样本数。

    参数：
        result: lm-eval 返回的完整结果。
        task: 请求执行的 task 或 group 名称。

    返回：
        规范指标映射和实际评测样本数。

    异常：
        ValueError: 结果缺少 task、数值指标、样本数或含重复指标名时抛出。
    """

    entry = _result_entry(result, task)
    metrics: dict[str, float] = {}
    for raw_name, value in entry.items():
        name = _metric_name(str(raw_name))
        if name is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if name in metrics:
            raise ValueError(f"lm-eval result contains duplicate metric {name!r}")
        metrics[name] = float(value)
    if not metrics:
        raise ValueError(f"lm-eval result does not contain metrics for {task!r}")
    return metrics, _sample_count(result, task, entry)


class LMEvalEvaluator:
    """使用一个 lm-eval task 或 group 重复评测不同模型状态。"""

    def __init__(
        self,
        tokenizer: Any,
        config: LMEvalConfig,
        *,
        runtime_config_path: str | Path | None = None,
    ) -> None:
        """保存 tokenizer、task 配置和可选 runtime 路径。

        参数：
            tokenizer: 与待评测模型匹配的 Hugging Face tokenizer。
            config: task、batch、上下文长度、样本限制和随机种子。
            runtime_config_path: 可选 runtime TOML；省略时使用项目默认配置。
        """

        self.tokenizer = tokenizer
        self.config = config
        self.runtime_config_path = runtime_config_path
        self._task_manager: Any | None = None
        self._task: Any | None = None
        self._samples: dict[str, list[int]] | None = None

    def _prepare_task(self) -> tuple[Any, Any]:
        """首次从本地缓存加载 task，后续复用同一对象。

        返回：
            lm-eval TaskManager 和请求的 task/group 对象。

        异常：
            KeyError: TaskManager 没有返回请求名称时抛出。
            ValueError: 非零题目起点用于 group 或超出评测集时抛出。
        """

        if self._task_manager is None:
            configure_runtime(self.runtime_config_path)
            from lm_eval.tasks import TaskManager

            task_manager = TaskManager()
            loaded = task_manager.load([self.config.task])
            task = loaded.get("groups", {}).get(self.config.task)
            if task is None:
                task = loaded.get("tasks", {}).get(self.config.task)
            if task is None:
                raise KeyError(f"lm-eval did not load task {self.config.task!r}")
            if self.config.sample_start_index:
                if self.config.task not in loaded.get("tasks", {}):
                    raise ValueError(
                        "sample_start_index requires a single task, not a group"
                    )
                count = len(task.eval_docs)
                start = self.config.sample_start_index
                if start >= count:
                    raise ValueError(
                        f"sample_start_index {start} is outside {count} evaluation samples"
                    )
                stop = (
                    count
                    if self.config.limit is None
                    else min(count, start + self.config.limit)
                )
                self._samples = {self.config.task: list(range(start, stop))}
            self._task_manager = task_manager
            self._task = task
        return self._task_manager, self._task

    def __call__(self, model: nn.Module) -> EvaluationResult:
        """使用已准备的 lm-eval task 评测当前模型状态。

        参数：
            model: 已经加载到单个目标设备的 Causal LM。

        返回：
            包含动态指标和实际样本数的通用评测结果。

        异常：
            RuntimeError: lm-eval 没有返回结果时抛出。
            ValueError: lm-eval 结果无法转换时抛出。
            ModuleNotFoundError: 当前环境没有安装 lm-eval 时抛出。
        """

        configure_runtime(self.runtime_config_path)
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        task_manager, task = self._prepare_task()
        runtime_device = str(next(model.parameters()).device)
        original_training = model.training
        model.eval()
        try:
            evaluator_model = HFLM(
                pretrained=model,
                tokenizer=self.tokenizer,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                device=runtime_device,
            )
            raw_result = lm_eval.simple_evaluate(
                model=evaluator_model,
                tasks=[task],
                task_manager=task_manager,
                num_fewshot=self.config.num_fewshot,
                limit=self.config.limit if self._samples is None else None,
                samples=self._samples,
                # lm-eval 的请求缓存键不含 samples，偏移评测须禁用读写以免串题。
                cache_requests=self._samples is None,
                bootstrap_iters=0,
                apply_chat_template=self.config.apply_chat_template,
                log_samples=False,
                random_seed=self.config.seed,
                numpy_random_seed=self.config.seed,
                torch_random_seed=self.config.seed,
                fewshot_random_seed=self.config.seed,
            )
        finally:
            model.train(original_training)

        if raw_result is None:
            raise RuntimeError("lm-eval returned no result")
        metrics, evaluated_examples = _extract_metrics(
            raw_result,
            self.config.task,
        )
        if self._samples is not None and evaluated_examples != len(
            self._samples[self.config.task]
        ):
            raise ValueError("evaluated sample count differs from requested sample range")
        fewshot = (
            "default"
            if self.config.num_fewshot is None
            else str(self.config.num_fewshot)
        )
        preprocessing = f"lm-eval-{fewshot}-shot"
        if self._samples is not None:
            start = self.config.sample_start_index
            preprocessing += f"-samples-{start}:{start + evaluated_examples}"
        evaluation_task = EvaluationTask(
            name=self.config.task,
            dataset=self.config.task,
            split="lm-eval",
            preprocessing=preprocessing,
            requested_metrics=tuple(metrics),
        )
        counts = raw_result.get("n-samples", {})
        originals = [entry.get("original") for entry in counts.values()]
        total_examples = (
            sum(originals)
            if originals and all(type(value) is int and value > 0 for value in originals)
            else None
        )
        return EvaluationResult(
            task=evaluation_task,
            metrics=metrics,
            evaluated_examples=evaluated_examples,
            total_examples=total_examples,
        )
