"""公开 qcomp 第一版的顶层接口。

本包集中导出 artifact、通用重建入口、MPO 表示工具、后端查询、模型层替换、模型级
压缩、敏感性分析与微调工作流和持久化函数，为调用者提供稳定入口。具体 Provider 保持隔离，
只有通过 registry 选择后才会被加载。

主要内容：
- ``load_compression_plan``、``compression_plan_to_dict``、``compression_plan_from_dict``：读写并验证压缩计划。
- ``TensorNetworkArtifact``：保存与具体张量网络表示无关的数据。
- ``reconstruct_tensor``：根据 artifact 的表示类型重建稠密张量。
- ``MPOSpec``：描述 MPO 的输入维度、输出维度和 TT ranks。
- ``get_backend``、``list_backends``：查询 Provider 与表示的已注册组合。
- ``ModelLoadConfig``、``CausalLMResources``：定义通用 Causal LM 加载接口。
- ``load_causal_lm``：加载模型和 tokenizer，并将模型放到目标设备。
- ``list_linears``、``list_tensor_network_linears``：列出稠密和张量网络 Linear。
- ``find_linear``、``replace_linear``、``restore_linear``：替换和恢复模型中的 Linear。
- ``CompressionTarget``、``CompressionPlan``：描述模型级压缩目标。
- ``compress_linear``、``compress_model``：执行单层或模型级压缩。
- ``SensitivityCase``、``SensitivityResult``：描述并汇总压缩敏感性实验。
- ``SensitivityExperimentConfig``、``run_sensitivity_experiment``：配置并执行逐层实验。
- ``analyze_sensitivity``：使用外部 backend 和 evaluator 执行敏感性分析。
- ``sensitivity_case_record``：将敏感性结果整理为实验记录字段。
- ``format_sensitivity_report``：生成并按需保存敏感性 Markdown 报告。
- ``metric_direction``、``resolve_metric_directions``：查询常用评测指标方向。
- ``DatasetSource``、``load_dataset_source``：统一描述和加载三类数据来源。
- ``DataLoaderConfig``、``build_causal_lm_dataloader``：构造通用训练 DataLoader。
- ``RuntimeConfig``、``configure_runtime``：读取默认 runtime.toml 并应用离线环境。
- ``log_event``：统一追加 JSON Lines 实验日志。
- ``ArtifactPaths``：配置项目运行产物的标准目录。
- ``TrainingConfig``、``TrainingObjective``、``TrainingResult``：定义通用训练接口。
- ``CausalLMObjective``、``train_causal_lm``：提供默认 loss 和公共训练循环。
- ``TensorNetworkFineTuneResult``：组合通用训练结果和张量网络特有产物。
- ``finetune_tensor_network_causal_lm``：只更新张量网络参数并导出 artifacts。
- ``InferenceConfig``、``InferenceResult``：配置正常生成并返回 token 与运行开销。
- ``infer_causal_lm``：执行完整 Causal LM 的正常自回归生成。
- ``make_mpo_artifact``、``parse_mpo_artifact``、``reconstruct_mpo``：转换和重建 MPO。
- ``save_artifact``、``load_artifact``：保存和加载通用 artifact。
"""

from .backends import get_backend, list_backends
from .data import (
    DataLoaderConfig,
    DatasetSource,
    HuggingFaceDatasetSource,
    JsonlDatasetSource,
    LocalDatasetSource,
    build_causal_lm_dataloader,
    load_dataset_source,
)
from .evaluation import (
    MetricDirection,
    metric_direction,
    resolve_metric_directions,
)
from .logging import log_event
from .model import (
    CausalLMResources,
    LinearReplacement,
    ModelLoadConfig,
    find_linear,
    list_linears,
    list_tensor_network_linears,
    load_causal_lm,
    replace_linear,
    restore_linear,
)
from .representations import (
    MPOSpec,
    TensorNetworkArtifact,
    make_mpo_artifact,
    parse_mpo_artifact,
    reconstruct_mpo,
    reconstruct_tensor,
)
from .runtime import RuntimeConfig, configure_runtime, load_runtime_config
from .storage import ArtifactPaths, load_artifact, save_artifact
from .training import (
    CausalLMObjective,
    TrainingConfig,
    TrainingObjective,
    TrainingResult,
    train_causal_lm,
)
from .workflows import (
    CompressionPlan,
    CompressionTarget,
    InferenceConfig,
    InferencePerformance,
    InferenceResult,
    LinearCompressionResult,
    ModelCompressionResult,
    ModelEvaluator,
    SensitivityCase,
    SensitivityExperimentConfig,
    run_sensitivity_experiment,
    SensitivityCaseResult,
    SensitivityResult,
    TensorNetworkFineTuneResult,
    analyze_sensitivity,
    compress_linear,
    format_sensitivity_report,
    sensitivity_case_record,
    compress_model,
    finetune_tensor_network_causal_lm,
    infer_causal_lm,
    restore_compressed_model,
)

from .workflows.compression_plan_io import (
    compression_plan_to_dict,
    compression_plan_from_dict,
    load_compression_plan,
)

__all__ = [
    "compression_plan_to_dict",
    "compression_plan_from_dict",
    "load_compression_plan",
    "ArtifactPaths",
    "CausalLMResources",
    "DataLoaderConfig",
    "DatasetSource",
    "HuggingFaceDatasetSource",
    "InferenceConfig",
    "InferencePerformance",
    "InferenceResult",
    "CompressionPlan",
    "CompressionTarget",
    "TrainingConfig",
    "CausalLMObjective",
    "MPOSpec",
    "JsonlDatasetSource",
    "LinearReplacement",
    "LocalDatasetSource",
    "LinearCompressionResult",
    "MetricDirection",
    "ModelCompressionResult",
    "ModelEvaluator",
    "SensitivityCase",
    "SensitivityExperimentConfig",
    "run_sensitivity_experiment",
    "SensitivityCaseResult",
    "SensitivityResult",
    "ModelLoadConfig",
    "TensorNetworkArtifact",
    "TrainingObjective",
    "TensorNetworkFineTuneResult",
    "TrainingResult",
    "RuntimeConfig",
    "analyze_sensitivity",
    "build_causal_lm_dataloader",
    "configure_runtime",
    "log_event",
    "find_linear",
    "finetune_tensor_network_causal_lm",
    "format_sensitivity_report",
    "sensitivity_case_record",
    "infer_causal_lm",
    "compress_linear",
    "compress_model",
    "get_backend",
    "list_backends",
    "list_linears",
    "list_tensor_network_linears",
    "load_artifact",
    "load_dataset_source",
    "load_runtime_config",
    "load_causal_lm",
    "make_mpo_artifact",
    "metric_direction",
    "parse_mpo_artifact",
    "reconstruct_mpo",
    "reconstruct_tensor",
    "replace_linear",
    "resolve_metric_directions",
    "restore_linear",
    "restore_compressed_model",
    "save_artifact",
    "train_causal_lm",
]
