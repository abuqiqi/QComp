"""公开与后端无关的压缩指标和性能计时接口。

本包统一导出参数量、张量字节数、重建质量指标和分解、推理、训练计时函数。
evaluation 只依赖公开 backend 与 artifact 接口，不访问具体 Provider 的内部实现。

主要内容：
- ``CompressionMetrics``、``compression_metrics``：计算规模、字节数和重建误差。
- ``TimingResult``：保存一组计时样本及其汇总统计。
- ``time_decomposition``：测量权重分解时间。
- ``time_inference``：测量模型层推理时间。
- ``time_training_step``：测量完整训练 step 时间。
"""

from .compression import CompressionMetrics, compression_metrics
from .performance import (
    TimingResult,
    time_decomposition,
    time_inference,
    time_training_step,
)

__all__ = [
    "CompressionMetrics",
    "TimingResult",
    "compression_metrics",
    "time_decomposition",
    "time_inference",
    "time_training_step",
]
