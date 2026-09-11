"""公开可直接调用的 qcomp 工作流。

本包把 model、representations、backends 和 training 层的基础能力组合成面向任务的
操作。当前提供单层与模型级压缩、压缩敏感性分析、已压缩 Causal LM 微调和正常生成流程。

主要内容：
- ``CompressionExecutionConfig``：组合分解、执行后端及分解精度。
- ``evaluate_compression_plans``：共享 baseline 评测多个方案与任务。
- ``load_compression_plan``、``compression_plan_to_dict``、``compression_plan_from_dict``：读写并验证压缩计划。
- ``CompressionTarget``、``CompressionPlan``：描述模型级压缩目标。
- ``LinearCompressionResult``：保存单层压缩产生的 artifact 和模型替换记录。
- ``ModelCompressionResult``：保存模型级压缩产生的逐层结果。
- ``compress_linear``：分解一个 Linear，并安装指定执行后端构造的压缩模型层。
- ``compress_model``、``restore_compressed_model``：批量压缩和恢复模型。
- ``SensitivityExperimentConfig``、``run_sensitivity_experiment``：配置并执行逐层实验。
- ``sensitivity_case_record``：将敏感性结果整理为实验记录字段。
- ``format_sensitivity_report``：生成并按需保存敏感性 Markdown 报告。
- ``TensorNetworkFineTuneResult``：组合通用训练结果和张量网络特有产物。
- ``finetune_tensor_network_causal_lm``：只更新张量网络参数并支持断点续训。
- ``InferenceConfig``、``InferenceResult``：定义正常生成配置和组合结果。
- ``infer_causal_lm``：生成 token 并记录时间、吞吐与峰值显存。
"""

from ..evaluation import MetricDirection
from .compress import (
    CompressionExecutionConfig,
    CompressionPlan,
    CompressionTarget,
    LinearCompressionResult,
    ModelCompressionResult,
    compress_linear,
    compress_model,
    restore_compressed_model,
)
from .compression_plan_io import (
    compression_plan_from_dict,
    compression_plan_to_dict,
    load_compression_plan,
)
from .evaluate import (
    CompressionEvaluationResult,
    CompressionPlanEvaluation,
    ModelEvaluator,
    TimedEvaluation,
    evaluate_compression_plans,
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
from .sensitivity import (
    SensitivityExperimentConfig,
    SensitivityExperimentResult,
    format_sensitivity_report,
    run_sensitivity_experiment,
    sensitivity_case_record,
)

__all__ = [
    "CompressionEvaluationResult",
    "CompressionExecutionConfig",
    "CompressionPlan",
    "CompressionPlanEvaluation",
    "CompressionTarget",
    "InferenceConfig",
    "InferencePerformance",
    "InferenceResult",
    "LinearCompressionResult",
    "MetricDirection",
    "ModelCompressionResult",
    "ModelEvaluator",
    "SensitivityExperimentConfig",
    "SensitivityExperimentResult",
    "TensorNetworkFineTuneResult",
    "TimedEvaluation",
    "compress_linear",
    "compress_model",
    "compression_plan_from_dict",
    "compression_plan_to_dict",
    "evaluate_compression_plans",
    "finetune_tensor_network_causal_lm",
    "format_sensitivity_report",
    "infer_causal_lm",
    "load_compression_plan",
    "read_sensitivity_results",
    "restore_compressed_model",
    "run_sensitivity_experiment",
    "sensitivity_case_record",
    "validate_sensitivity_results",
    "write_sensitivity_results",
]

from .sensitivity_io import (
    read_sensitivity_results,
    validate_sensitivity_results,
    write_sensitivity_results,
)
