# 扩展指南

本指南说明如何在现有模块边界内增加张量网络表示和训练方法。当前实现以 MPO 和 Causal LM objective 为参考；Tucker、Cayley 等名称仅用于说明扩展职责。

## 新增张量网络表示

这里的表示指 MPO、Tucker 等数学结构；浮点 dtype 由张量保存，改变 dtype 不需要新增表示或后端。

1. **定义结构和格式。** 在 `src/qcomp/representations/<表示>.py` 定义结构配置、维度与布局校验、artifact 构造和解析、稠密重建。复用 [TensorNetworkArtifact](../src/qcomp/representations/artifact.py)，用 `representation` 标识结构，`metadata` 保存结构元数据，`tensors` 保存命名参数。明确张量顺序、形状、dtype/device 约束和支持的格式版本；参考 [mpo.py](../src/qcomp/representations/mpo.py)。
2. **注册重建与公共入口。** 在 [representations registry](../src/qcomp/representations/registry.py) 的 `_RECONSTRUCTORS` 添加表示名到模块、函数名的映射，并在对应 `__init__.py` 导出公开对象；需要顶层调用的接口再加入 `qcomp/__init__.py`。这样 `reconstruct_tensor()` 和压缩误差评测可沿用通用入口。
3. **定义可执行层。** 具体模型层继承 [TensorNetworkLinear](../src/qcomp/nn/base.py)，实现 forward、最新 artifact 导出，以及必要的 `close()`。多个后端确有共享输入处理或导出行为时，才在 `nn/<表示>.py` 提取中间基类。模型层使用 PyTorch 参数注册机制，以便设备迁移、训练参数选择和 checkpoint 工作。
4. **实现实际后端。** 在 `backends/<provider>/<表示>.py` 实现 `TensorNetworkBackend[Spec]`，声明准确的 capabilities、probe 和版本查询；实现支持的分解及模型层构造，使用统一 artifact 交换数据。可选库延迟导入，在 [backend registry](../src/qcomp/backends/registry.py) 的 `_BACKENDS` 注册真实组合。计算库适配细节见 [backends](../src/qcomp/backends/README.md#新增计算后端)。
5. **接入 workflow。** 用 `CompressionTarget(module_path, representation, spec)` 构造计划，通过按表示名索引的后端映射交给 `compress_model()` 或 `evaluate_compression_plans()`。保留目标 Linear 的输入输出形状及设备、dtype 语义；混合表示复用已有计划和恢复逻辑。实验脚本只补参数和表示专属目标构造。
6. **验证。** 使用小型确定性张量检查非法格式、稠密重建与 forward 数值、实际支持的梯度、artifact 保存加载和后端互通；检查多层计划失败恢复、模型压缩指标与可选依赖懒加载。计时测量复用 evaluation 接口。

## 新增训练方法

1. **确定参数与模型组件。** workflow 安装所需组件并明确列出可训练参数名称。压缩层复用 `TensorNetworkLinear`；例如新的 Adapter 使用普通 PyTorch 参数注册。参数选择属于 workflow，公共 trainer 不识别具体表示或 Adapter。
2. **实现 objective。** 按 [TrainingObjective](../src/qcomp/training/objectives.py) 实现 `metadata` 和 `__call__(model, batch)`。batch 已移动到训练设备，返回可反向传播的标量 Tensor；metadata 保存恢复时必须一致的目标配置。蒸馏 objective 由调用方提供 teacher 并管理其设备和冻结状态，不假设 teacher 会随学生 checkpoint 保存。
3. **编排训练。** 在 `workflows/` 增加具体方法入口，负责构造 objective、选择参数、调用 `train_causal_lm()` 和导出该方法产物；复用 [finetune workflow](../src/qcomp/workflows/finetune.py) 的职责划分。共享 objective 放在 training 层，实验专属策略留在调用方。
4. **约定保存与恢复。** [trainer](../src/qcomp/training/trainer.py) 保存选定参数、优化器、调度器、训练位置、配置、objective metadata 和随机数状态，不保存冻结参数或任意外部对象。恢复前重建相同参数结构、冻结权重及所需外部资源，并保持参数名称、训练配置、objective metadata、总步数和 DataLoader 长度一致；数据顺序应可复现。张量网络 artifact 由 workflow 导出，其他组件需要明确各自产物的保存和加载接口。
5. **按需要扩展 trainer。** 新方法能用参数选择和 loss 表达时，复用现有 AdamW、调度、梯度累积和 checkpoint。只有需要不同优化器、调度或更新步骤且现有接口无法表达时，才修改公共训练机制及对应配置、恢复校验；不复制训练循环。
6. **验证。** 检查仅选定参数更新、冻结参数不变、objective 产生有效梯度、产物可加载；比较连续训练与中间 checkpoint 恢复结果，并验证配置不匹配时明确失败。涉及新后端或数学运算时增加数值与梯度测试。

## 文档与入口

新接口在所属模块 `__init__.py` 导出，按需要加入顶层导出。模块 README 保留职责和规范用法，扩展步骤集中维护在本指南；脚本使用现有公共入口，实验命令记录在[实验指南](experiments.md)。
