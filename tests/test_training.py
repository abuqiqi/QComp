"""验证通用 Causal LM 训练接口与 objective 扩展边界。

本模块使用普通 PyTorch 模型和自定义 loss objective，确认 training 层只更新调用者
指定的参数，并将 objective 元数据写入 checkpoint。它还静态检查通用训练代码不依赖
张量网络表示、模型层或计算后端。

主要内容：
- ``RegressionModel``：提供两个可独立选择参数的最小模型。
- ``ScaledMSEObjective``：验证自定义 objective 调用和元数据。
- ``TrainingTests``：覆盖参数选择、checkpoint objective 校验和依赖边界。
"""

import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from qcomp import TrainingConfig, train_causal_lm


class RegressionModel(nn.Module):
    """提供两个连续线性层的最小训练模型。"""

    def __init__(self) -> None:
        """创建确定初始化的一维线性模型。"""

        super().__init__()
        self.selected = nn.Linear(1, 1, bias=False)
        self.frozen = nn.Linear(1, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """依次执行两个线性层。

        参数：
            inputs: 形状为 [batch, 1] 的输入。

        返回：
            形状与输入相同的预测结果。
        """

        return self.frozen(self.selected(inputs))


class ScaledMSEObjective:
    """使用可配置比例计算均方误差。"""

    def __init__(self, scale: float) -> None:
        """保存 loss 比例。

        参数：
            scale: 乘到均方误差上的比例。
        """

        self.scale = scale

    @property
    def metadata(self) -> dict[str, Any]:
        """返回用于 checkpoint 匹配的 objective 配置。"""

        return {"name": "scaled_mse", "scale": self.scale}

    def __call__(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """计算模型预测与目标之间的缩放均方误差。

        参数：
            model: 当前训练模型。
            batch: 包含 inputs 和 targets 的训练 batch。

        返回：
            标量均方误差。
        """

        prediction = model(batch["inputs"])
        return self.scale * nn.functional.mse_loss(prediction, batch["targets"])


def _make_loader() -> DataLoader[Any]:
    """创建确定性回归训练数据。"""

    examples = [
        {"inputs": torch.tensor([1.0]), "targets": torch.tensor([0.0])},
        {"inputs": torch.tensor([2.0]), "targets": torch.tensor([0.0])},
    ]
    return DataLoader(
        examples,
        batch_size=1,
        generator=torch.Generator().manual_seed(7),
    )


class TrainingTests(unittest.TestCase):
    """验证通用训练循环的公开扩展接口。"""

    def test_train_causal_lm_updates_only_named_parameters(self) -> None:
        """使用自定义 objective 只更新明确指定的模型参数。"""

        torch.manual_seed(5)
        model = RegressionModel()
        selected_before = model.selected.weight.detach().clone()
        frozen_before = model.frozen.weight.detach().clone()
        config = TrainingConfig(
            max_steps=1,
            learning_rate=1e-1,
            warmup_ratio=0.0,
            device="cpu",
            bf16_autocast=False,
        )
        with TemporaryDirectory() as temporary:
            result = train_causal_lm(
                model,
                _make_loader(),
                ("selected.weight",),
                ScaledMSEObjective(1.0),
                config,
                temporary,
            )

        self.assertEqual(result.parameter_names, ("selected.weight",))
        self.assertFalse(torch.equal(model.selected.weight, selected_before))
        torch.testing.assert_close(model.frozen.weight, frozen_before)
        self.assertTrue(model.selected.weight.requires_grad)
        self.assertFalse(model.frozen.weight.requires_grad)

    def test_resume_rejects_different_objective_metadata(self) -> None:
        """拒绝使用不同 objective 配置恢复已有 checkpoint。"""

        config = TrainingConfig(
            max_steps=2,
            learning_rate=1e-1,
            warmup_ratio=0.0,
            save_steps=1,
            device="cpu",
            bf16_autocast=False,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(5)
            source_model = RegressionModel()
            train_causal_lm(
                source_model,
                _make_loader(),
                ("selected.weight",),
                ScaledMSEObjective(1.0),
                config,
                root / "source",
            )

            torch.manual_seed(5)
            resumed_model = RegressionModel()
            with self.assertRaisesRegex(ValueError, "objective"):
                train_causal_lm(
                    resumed_model,
                    _make_loader(),
                    ("selected.weight",),
                    ScaledMSEObjective(2.0),
                    config,
                    root / "resumed",
                    resume_from=root / "source" / "checkpoint-1.pt",
                )

    def test_training_layer_has_no_tensor_network_dependency(self) -> None:
        """确认通用训练模块不导入表示、模型层或 backend。"""

        training_root = Path(__file__).parents[1] / "src" / "qcomp" / "training"
        forbidden = {"representations", "backends", "nn", "model"}
        imported: set[str] = set()
        for path in training_root.glob("*.py"):
            tree = ast.parse(path.read_text())
            imported.update(
                node.module.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
        self.assertTrue(forbidden.isdisjoint(imported))


if __name__ == "__main__":
    unittest.main()
