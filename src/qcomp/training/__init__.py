"""公开与具体微调方法无关的训练接口。

本包提供统一 Causal LM 训练循环和可插拔 loss objective。workflows 负责选择参数、
组装 objective 并导出方法特有产物，training 不依赖张量网络表示或计算后端。

主要内容：
- ``TrainingObjective``、``CausalLMObjective``：定义并实现 loss 计算接口。
- ``TrainingConfig``、``TrainingResult``：定义公共训练配置和结果。
- ``train_causal_lm``：执行训练、checkpoint 和断点续训。
"""

from .objectives import CausalLMObjective, TrainingObjective
from .trainer import TrainingConfig, TrainingResult, train_causal_lm

__all__ = [
    "CausalLMObjective",
    "TrainingConfig",
    "TrainingObjective",
    "TrainingResult",
    "train_causal_lm",
]
