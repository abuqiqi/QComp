"""验证单个无 bias Linear 的查找、替换和恢复。

本模块使用小型 PyTorch 模型连接 model、backend 和 nn 三层，确认已经构造的张量
网络模型层可以被批量发现并按路径安装，保持完整模型 forward，并恢复为同一个原始
Linear。

主要内容：
- ``ToyModel``：包含嵌套无 bias Linear 的最小测试模型。
- ``ModelTests``：覆盖层列出、查找、替换、推理、恢复和 bias 边界。
"""

import unittest

import torch
from torch import Tensor, nn

from qcomp import (
    MPOSpec,
    find_linear,
    get_backend,
    list_linears,
    replace_linear,
    restore_linear,
)


class ToyModel(nn.Module):
    """提供一个带嵌套 Linear 的最小 PyTorch 模型。"""

    def __init__(self, *, bias: bool = False) -> None:
        """创建确定形状的测试模型。

        参数：
            bias: 测试 Linear 是否包含 bias。
        """

        super().__init__()
        self.block = nn.ModuleDict({"projection": nn.Linear(4, 4, bias=bias)})

    def forward(self, inputs: Tensor) -> Tensor:
        """通过嵌套 Linear 计算输出。

        参数：
            inputs: 最后一维为 4 的输入张量。

        返回：
            嵌套 Linear 的输出。
        """

        return self.block["projection"](inputs)


class ModelTests(unittest.TestCase):
    """验证模型中单个 Linear 的可逆替换。"""

    def test_list_linears_returns_only_no_bias_layers(self) -> None:
        """按模块路径列出无 bias Linear，并跳过带 bias 的层。"""

        model = nn.ModuleDict(
            {
                "without_bias": nn.Linear(4, 4, bias=False),
                "with_bias": nn.Linear(4, 4, bias=True),
            }
        )

        linears = list_linears(model)

        self.assertEqual(tuple(name for name, _ in linears), ("without_bias",))
        self.assertIs(linears[0][1], model["without_bias"])

    def test_replace_and_restore_nested_linear(self) -> None:
        """安装 native MPO 层，完成 forward 后恢复原始 Linear。"""

        torch.manual_seed(11)
        model = ToyModel().to(torch.float64)
        inputs = torch.randn(3, 4, dtype=torch.float64)
        target = "block.projection"
        original = find_linear(model, target)
        expected = model(inputs)

        spec = MPOSpec.full_rank((2, 2), (2, 2))
        backend = get_backend("native", "mpo")
        artifact = backend.decompose(original.weight.detach(), spec)
        compressed = backend.build_linear(artifact, trainable=False)

        record = replace_linear(model, target, compressed)
        self.assertIs(model.get_submodule(target), compressed)
        torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-8)

        restore_linear(model, record)
        self.assertIs(model.get_submodule(target), original)
        torch.testing.assert_close(model(inputs), expected)

    def test_find_linear_rejects_bias(self) -> None:
        """拒绝当前压缩模型层尚不支持的 bias Linear。"""

        model = ToyModel(bias=True)
        with self.assertRaises(ValueError):
            find_linear(model, "block.projection")


if __name__ == "__main__":
    unittest.main()
