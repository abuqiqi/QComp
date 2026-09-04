"""验证压缩 Causal LM 的模型级微调和断点续训。

本模块使用包含两个 native MPO Linear 的小型语言模型和确定性 DataLoader，检查训练
循环只更新张量网络参数、导出每层 artifact，并能从中间 checkpoint 恢复到与连续
训练相同的状态。

主要内容：
- ``TinyCausalLM``：返回 Causal LM 标量 loss 的小型测试模型。
- ``FineTuneTests``：覆盖参数冻结、多层训练、checkpoint 恢复和缺少压缩层的情况。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from qcomp import (
    TrainingConfig,
    MPOSpec,
    compress_linear,
    finetune_tensor_network_causal_lm,
    get_backend,
)


class TinyCausalLM(nn.Module):
    """提供两个可压缩 Linear 和 Causal LM loss。"""

    def __init__(self) -> None:
        """创建词表大小为 8、隐藏维度为 4 的小型模型。"""

        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(4, 8, bias=False)

    def forward(self, input_ids: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """计算 token logits 和交叉熵损失。

        参数：
            input_ids: 形状为 [batch, sequence] 的 token IDs。
            labels: 与输入形状相同的目标 token IDs。

        返回：
            包含标量 loss 和 logits 的映射。
        """

        hidden = self.embedding(input_ids)
        hidden = self.dropout(torch.tanh(self.hidden(hidden)))
        logits = self.output(hidden)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        return {"loss": loss, "logits": logits}


def _make_loader() -> DataLoader[Any]:
    """创建顺序固定且不消耗全局随机数状态的训练 DataLoader。"""

    examples = [
        {
            "input_ids": torch.tensor([0, 1, 2]),
            "labels": torch.tensor([1, 2, 3]),
        },
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "labels": torch.tensor([4, 5, 6]),
        },
        {
            "input_ids": torch.tensor([6, 7, 0]),
            "labels": torch.tensor([7, 0, 1]),
        },
        {
            "input_ids": torch.tensor([2, 4, 6]),
            "labels": torch.tensor([3, 5, 7]),
        },
    ]
    generator = torch.Generator().manual_seed(31)
    return DataLoader(examples, batch_size=1, shuffle=False, generator=generator)


def _make_compressed_model() -> TinyCausalLM:
    """创建具有相同初始参数和两个可训练 MPO Linear 的模型。"""

    torch.manual_seed(17)
    model = TinyCausalLM().to(torch.float64)
    backend = get_backend("native", "mpo")
    compress_linear(
        model,
        "hidden",
        MPOSpec.full_rank((2, 2), (2, 2)),
        decomposition_backend=backend,
        execution_backend=backend,
        trainable=True,
    )
    compress_linear(
        model,
        "output",
        MPOSpec.full_rank((2, 4), (2, 2)),
        decomposition_backend=backend,
        execution_backend=backend,
        trainable=True,
    )
    return model


class FineTuneTests(unittest.TestCase):
    """验证模型级张量网络参数微调。"""

    def test_finetune_updates_only_tensor_network_parameters(self) -> None:
        """训练两个 MPO Linear，同时保持普通模型参数不变。"""

        model = _make_compressed_model()
        dense_before = model.embedding.weight.detach().clone()
        cores_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if ".cores." in name
        }
        config = TrainingConfig(
            max_steps=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-2,
            warmup_ratio=0.5,
            save_steps=1,
            device="cpu",
            bf16_autocast=False,
        )

        with TemporaryDirectory() as temporary:
            result = finetune_tensor_network_causal_lm(
                model,
                _make_loader(),
                config,
                Path(temporary),
            )

            self.assertEqual(result.training.global_step, 2)
            self.assertEqual(len(result.training.losses), 4)
            self.assertEqual(result.module_paths, ("hidden", "output"))
            self.assertEqual(tuple(result.tn_artifacts), ("hidden", "output"))
            self.assertTrue(result.training.checkpoint_path.is_file())
            self.assertTrue((Path(temporary) / "checkpoint-1.pt").is_file())

        torch.testing.assert_close(model.embedding.weight, dense_before)
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertTrue(
            any(
                not torch.equal(value, dict(model.named_parameters())[name])
                for name, value in cores_before.items()
            )
        )
        self.assertEqual(
            result.training.trainable_parameters,
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )

    def test_resume_matches_continuous_training(self) -> None:
        """从第一个优化步恢复，并得到与连续训练相同的最终 cores。"""

        config = TrainingConfig(
            max_steps=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-2,
            warmup_ratio=0.5,
            save_steps=1,
            seed=23,
            device="cpu",
            bf16_autocast=False,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_model = _make_compressed_model()
            continuous = finetune_tensor_network_causal_lm(
                continuous_model,
                _make_loader(),
                config,
                root / "continuous",
            )

            resumed_model = _make_compressed_model()
            resumed = finetune_tensor_network_causal_lm(
                resumed_model,
                _make_loader(),
                config,
                root / "resumed",
                resume_from=root / "continuous" / "checkpoint-1.pt",
            )

        self.assertEqual(
            resumed.training.global_step,
            continuous.training.global_step,
        )
        self.assertEqual(len(resumed.training.losses), 2)
        for path, expected in continuous.tn_artifacts.items():
            actual = resumed.tn_artifacts[path]
            self.assertEqual(tuple(actual.tensors), tuple(expected.tensors))
            for name, tensor in expected.tensors.items():
                torch.testing.assert_close(actual.tensors[name], tensor)

    def test_finetune_requires_trainable_tensor_network_parameters(self) -> None:
        """拒绝没有可训练张量网络参数的普通模型。"""

        model = TinyCausalLM()
        config = TrainingConfig(
            max_steps=1,
            device="cpu",
            bf16_autocast=False,
        )
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "TensorNetworkLinear"):
                finetune_tensor_network_causal_lm(
                    model,
                    _make_loader(),
                    config,
                    temporary,
                )


if __name__ == "__main__":
    unittest.main()
