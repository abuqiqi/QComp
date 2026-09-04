"""使用 lm-eval 对完整 Causal LM 执行 MMLU 多选题评测。

本模块接收已经加载到单个设备的模型和 tokenizer，由 lm-eval 负责读取 MMLU 缓存、
构造 few-shot prompt、计算选项得分并汇总总体准确率。输出转换为 evaluation 层统一的
`EvaluationResult`；模型加载、缓存下载和敏感性分析循环不在本模块中实现。

主要内容：
- ``MMLUEvaluationConfig``：定义执行参数，并生成对应的通用评测任务。
- ``evaluate_mmlu``：调用 lm-eval 并返回 MMLU 总体 accuracy。
- ``_extract_mmlu_metrics``：从 lm-eval 结果提取总体准确率和有效样本数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from .task import EvaluationResult, EvaluationTask


@dataclass(frozen=True)
class MMLUEvaluationConfig:
    """定义 lm-eval MMLU 评测配置。"""

    num_fewshot: int = 5
    batch_size: int = 8
    max_length: int = 4096
    limit: int | float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """确认 few-shot、batch、上下文长度和样本限制有效。

        异常：
            ValueError: 任一数值配置不符合 MMLU 评测约定时抛出。
        """

        if self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
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

    @property
    def task(self) -> EvaluationTask:
        """返回与当前 few-shot 配置对应的通用 MMLU 任务描述。"""

        return EvaluationTask(
            name="mmlu",
            dataset="cais/mmlu",
            split="test",
            preprocessing=f"lm-eval-{self.num_fewshot}-shot",
            requested_metrics=("accuracy",),
        )


def _extract_mmlu_metrics(result: Mapping[str, Any]) -> tuple[float, int]:
    """从 lm-eval 返回值提取 MMLU 总体 accuracy 和样本数。

    参数：
        result: `lm_eval.simple_evaluate()` 返回的映射。

    返回：
        总体 accuracy 和有效评测样本数。

    异常：
        ValueError: 返回值缺少数值 accuracy 或正样本数时抛出。
    """

    try:
        aggregate = result["results"]["mmlu"]
        accuracy = float(aggregate["acc,none"])
        evaluated_examples = int(aggregate["sample_count"]["acc,none"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "lm-eval result does not contain MMLU accuracy and sample count"
        ) from error
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("lm-eval MMLU accuracy must be between zero and one")
    if evaluated_examples <= 0:
        raise ValueError("lm-eval MMLU sample count must be positive")
    return accuracy, evaluated_examples


def evaluate_mmlu(
    model: nn.Module,
    tokenizer: Any,
    config: MMLUEvaluationConfig,
) -> EvaluationResult:
    """使用 lm-eval 计算完整 Causal LM 的 MMLU 总体准确率。

    参数：
        model: 已经加载到单个目标设备的 Causal LM。
        tokenizer: 与模型匹配的 Hugging Face tokenizer。
        config: few-shot、batch、上下文长度、样本限制和随机种子。

    返回：
        包含 MMLU accuracy 和有效样本数的统一评测结果。

    异常：
        RuntimeError: lm-eval 没有返回评测结果时抛出。
        ValueError: lm-eval 结果缺少总体 accuracy 或样本数时抛出。
        ModuleNotFoundError: 当前环境没有安装 lm-eval 时抛出。
    """

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    runtime_device = str(next(model.parameters()).device)
    original_training = model.training
    model.eval()
    try:
        evaluator_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=runtime_device,
        )
        raw_result = lm_eval.simple_evaluate(
            model=evaluator_model,
            tasks=["mmlu"],
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            bootstrap_iters=0,
            apply_chat_template=False,
            log_samples=False,
            random_seed=config.seed,
            numpy_random_seed=config.seed,
            torch_random_seed=config.seed,
            fewshot_random_seed=config.seed,
        )
    finally:
        model.train(original_training)

    if raw_result is None:
        raise RuntimeError("lm-eval returned no MMLU result")
    accuracy, evaluated_examples = _extract_mmlu_metrics(raw_result)
    return EvaluationResult(
        task=config.task,
        metrics={"accuracy": accuracy},
        evaluated_examples=evaluated_examples,
    )
