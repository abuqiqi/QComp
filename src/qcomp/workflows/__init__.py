"""公开可直接调用的 qcomp 工作流。

本包把 model、representations、backends 和 training 层的基础能力组合成面向任务的
操作。当前提供单层与模型级压缩、压缩敏感性分析、已压缩 Causal LM 微调和正常生成流程。

主要内容：
- ``CompressionTarget``、``CompressionPlan``：描述模型级压缩目标。
- ``LinearCompressionResult``：保存单层压缩产生的 artifact 和模型替换记录。
- ``ModelCompressionResult``：保存模型级压缩产生的逐层结果。
- ``compress_linear``：分解一个 Linear，并安装指定执行后端构造的压缩模型层。
- ``compress_model``、``restore_compressed_model``：批量压缩和恢复模型。
- ``SensitivityCase``、``SensitivityResult``：描述敏感性实验及其汇总结果。
- ``SensitivityExperimentConfig``、``run_sensitivity_experiment``：配置并执行逐层实验。
- ``analyze_sensitivity``：使用外部 backend 和 evaluator 执行压缩敏感性分析。
- ``sensitivity_case_record``：将敏感性结果整理为实验记录字段。
- ``format_sensitivity_report``：生成并按需保存敏感性 Markdown 报告。
- ``TensorNetworkFineTuneResult``：组合通用训练结果和张量网络特有产物。
- ``finetune_tensor_network_causal_lm``：只更新张量网络参数并支持断点续训。
- ``InferenceConfig``、``InferenceResult``：定义正常生成配置和组合结果。
- ``infer_causal_lm``：生成 token 并记录时间、吞吐与峰值显存。
"""

from .compress import (
    CompressionPlan,
    CompressionTarget,
    LinearCompressionResult,
    ModelCompressionResult,
    compress_linear,
    compress_model,
    restore_compressed_model,
)
from .finetune import (
    TensorNetworkFineTuneResult,
    finetune_tensor_network_causal_lm,
)
from .inference import (
    InferenceConfig,
    InferencePerformance,
    InferenceResult,
    infer_causal_lm,
)
from ..evaluation import MetricDirection
from .sensitivity import (
    ModelEvaluator,
    SensitivityCase,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
    SensitivityCaseResult,
    SensitivityResult,
    analyze_sensitivity,
    format_sensitivity_report,
    sensitivity_case_record,
)

__all__ = [
    "InferenceConfig",
    "InferencePerformance",
    "InferenceResult",
    "CompressionPlan",
    "CompressionTarget",
    "LinearCompressionResult",
    "ModelCompressionResult",
    "MetricDirection",
    "ModelEvaluator",
    "SensitivityCase",
    "SensitivityExperimentConfig",
    "run_sensitivity_experiment",
    "SensitivityCaseResult",
    "SensitivityResult",
    "TensorNetworkFineTuneResult",
    "analyze_sensitivity",
    "compress_linear",
    "compress_model",
    "finetune_tensor_network_causal_lm",
    "format_sensitivity_report",
    "sensitivity_case_record",
    "infer_causal_lm",
    "restore_compressed_model",
]
