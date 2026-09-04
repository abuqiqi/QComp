# training

`training` 提供 Causal LM 训练机制。上层 `workflows` 选择需要
更新的参数并构造 loss objective；本目录统一处理训练循环、优化器、学习率调度和
checkpoint，不依赖张量网络表示或具体 backend。

## 目录结构

```text
training/
├── README.md       # 通用训练接口、调用关系和扩展方式
├── __init__.py     # 导出公共训练对象
├── objectives.py  # loss objective 接口和默认 Causal LM objective
└── trainer.py     # 训练循环、配置、结果和 checkpoint
```

## 主要公共对象

- `TrainingConfig`：配置 epoch、step、AdamW、梯度累积、学习率调度和 checkpoint。
- `TrainingObjective`：约定 objective 的 `metadata` 和 loss 调用接口。
- `CausalLMObjective`：调用 `model(**batch)` 并读取模型输出中的 `loss`。
- `TrainingResult`：返回训练步数、loss、参数量、参数名称和 checkpoint 路径。
- `train_causal_lm()`：只更新调用者明确指定的参数，并执行统一训练流程。

`TrainingObjective.metadata` 会保存在 checkpoint 中。恢复训练时，参数名称、objective
配置、训练配置和 DataLoader 长度必须与保存时一致。

## 与 workflows 的关系

```text
finetune_tensor_network_causal_lm()
  → 选择 TensorNetworkLinear 参数
  → 创建 CausalLMObjective
  → train_causal_lm()
  → 导出 TensorNetworkArtifact
```

`train_causal_lm()` 不查找 MPO cores，也不导出 artifact。这样以后增加 Cayley
微调时，只需由新的 workflow 选择 Cayley Adapter 参数并传入蒸馏 objective，公共
训练循环和 checkpoint 不需要重复实现。

## 通用调用

```python
from qcomp import CausalLMObjective, TrainingConfig, train_causal_lm

result = train_causal_lm(
    model,
    train_dataloader,
    trainable_parameter_names=("adapter.parameters",),
    objective=CausalLMObjective(),
    config=TrainingConfig(max_steps=100, device="cuda:0"),
    output_dir="artifacts/checkpoints/run-1",
)
```

自定义 objective 实现 `metadata` 和 `__call__()` 即可。`__call__()` 接收模型及已经
移动到训练设备的 batch，并返回用于反向传播的标量 Tensor。
