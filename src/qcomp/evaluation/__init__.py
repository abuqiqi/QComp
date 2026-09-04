"""公开与后端无关的压缩指标和性能计时接口。

本包统一导出参数量、张量字节数、重建质量指标、Causal LM 质量指标和分解、推理、
训练计时函数。evaluation 不访问具体 Provider 的内部实现。

主要内容：
- ``CompressionMetrics``、``compression_metrics``：计算单张量规模和重建误差。
- ``ModelCompressionMetrics``、``model_compression_metrics``：计算完整模型压缩率。
- ``EvaluationTask``：描述数据集、split、预处理方式和请求的指标名称。
- ``EvaluationResult``：保存任务、动态指标和统计范围。
- ``evaluate_causal_lm``：评测完整模型的 loss 和 perplexity。
- ``MMLUEvaluationConfig``、``evaluate_mmlu``：通过 lm-eval 评测 MMLU accuracy。
- ``TimingResult``：保存一组计时样本及其汇总统计。
- ``time_decomposition``：测量权重分解时间。
- ``time_inference``：测量模型层推理时间。
- ``time_training_step``：测量完整训练 step 时间。
"""

from .compression import (
    CompressionMetrics,
    ModelCompressionMetrics,
    compression_metrics,
    model_compression_metrics,
)
from .mmlu import MMLUEvaluationConfig, evaluate_mmlu
from .performance import (
    TimingResult,
    time_decomposition,
    time_inference,
    time_training_step,
)
from .quality import evaluate_causal_lm
from .task import EvaluationResult, EvaluationTask

__all__ = [
    "EvaluationResult",
    "EvaluationTask",
    "MMLUEvaluationConfig",
    "CompressionMetrics",
    "ModelCompressionMetrics",
    "TimingResult",
    "compression_metrics",
    "model_compression_metrics",
    "evaluate_causal_lm",
    "evaluate_mmlu",
    "time_decomposition",
    "time_inference",
    "time_training_step",
]
