"""公开与后端无关的压缩指标和性能计时接口。

本包统一导出参数量、张量字节数、重建质量指标、lm-eval benchmark 和分解、推理、
训练计时函数。evaluation 不访问具体 Provider 的内部实现。

主要内容：
- ``EvaluationTaskConfig``：组合任务执行参数与关注指标方向。
- ``CompressionMetrics``、``compression_metrics``：计算单张量规模和重建误差。
- ``ModelCompressionMetrics``、``model_compression_metrics``：计算完整模型压缩率。
- ``EvaluationTask``：描述数据集、split、预处理方式和请求的指标名称。
- ``EvaluationResult``：保存任务、动态指标和统计范围。
- ``LMEvalConfig``：配置任一 lm-eval task 或 group。
- ``LMEvalEvaluator``：复用 lm-eval 任务评测不同模型状态。
- ``lm_eval_dataset_size``：读取完整评测集样本数。
- ``MetricDirection``：限定指标优化方向。
- ``metric_direction``、``resolve_metric_directions``：查询常用指标方向。
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
from .lm_eval import (
    EvaluationTaskConfig,
    LMEvalConfig,
    LMEvalEvaluator,
    lm_eval_dataset_size,
)
from .metrics import MetricDirection, metric_direction, resolve_metric_directions
from .performance import (
    TimingResult,
    time_decomposition,
    time_inference,
    time_training_step,
)
from .task import EvaluationResult, EvaluationTask

__all__ = [
    "EvaluationTaskConfig",
    "EvaluationResult",
    "EvaluationTask",
    "LMEvalConfig",
    "LMEvalEvaluator",
    "lm_eval_dataset_size",
    "MetricDirection",
    "CompressionMetrics",
    "ModelCompressionMetrics",
    "TimingResult",
    "compression_metrics",
    "model_compression_metrics",
    "metric_direction",
    "resolve_metric_directions",
    "time_decomposition",
    "time_inference",
    "time_training_step",
]
