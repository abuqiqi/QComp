# training

`training` 提供 Causal LM 训练机制。上层 `workflows` 选择需要更新的参数并构造 loss objective；本目录统一处理训练循环、优化器、学习率调度和 checkpoint，不依赖张量网络表示或具体 backend。

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

`TrainingObjective.metadata` 会保存在 checkpoint 中。恢复训练时，参数名称、objective 配置、训练配置和 DataLoader 长度必须与保存时一致。

## 与 workflows 的关系

```text
finetune_tensor_network_causal_lm()
  → 选择 TensorNetworkLinear 参数
  → 创建 CausalLMObjective
  → train_causal_lm()
  → 导出 TensorNetworkArtifact
```

参数选择和 artifact 导出由 workflow 负责；新增 objective 或微调方法见[扩展指南](../../../docs/extending.md#新增训练方法)。

## 通用调用

下面用小型 CPU 模型演示完整调用；真实 Causal LM 可使用同一入口和已 tokenize 的 DataLoader。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from qcomp import CausalLMObjective, TrainingConfig, train_causal_lm

class TinyLM(nn.Module):
    """提供返回标量 loss 的最小训练模型。"""

    def __init__(self):
        """创建一个可训练参数。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids):
        """根据输入 input_ids 计算演示损失。"""
        return {"loss": (input_ids.float() * self.weight).square().mean()}

model = TinyLM()
train_dataloader = DataLoader([{"input_ids": torch.tensor([1, 2])}])
result = train_causal_lm(
    model,
    train_dataloader,
    trainable_parameter_names=("weight",),
    objective=CausalLMObjective(),
    config=TrainingConfig(max_steps=2, device="cpu"),
    output_dir="artifacts/checkpoints/run-1",
)
```

自定义 objective 实现 `metadata` 和 `__call__()` 即可。`__call__()` 接收模型及已经移动到训练设备的 batch，并返回用于反向传播的标量 Tensor。
