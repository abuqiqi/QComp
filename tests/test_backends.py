"""验证所有 MPO 后端的公共接口和数值行为。

本模块通过规范 artifact 比较 native、TensorLy、torchTT 和 cuTensorNet，覆盖后端
接口、模型层接口、分解、推理、梯度、参数更新和可选 Provider 延迟导入。

主要内容：
- ``BackendTests``：测试 registry、四种 MPO 适配器及其模型层支持的完整路径。
"""

import importlib.util
import subprocess
import sys
import unittest

import torch
from torch.nn import functional

from qcomp import MPOSpec, get_backend, list_backends, reconstruct_mpo
from qcomp.backends import TensorNetworkBackend
from qcomp.nn import TensorNetworkLinear


class BackendTests(unittest.TestCase):
    """通过同一规范 MPO artifact 比较所有后端。"""

    def setUp(self) -> None:
        """创建后端测试共享的确定性满 rank 数据。"""

        torch.manual_seed(7)
        self.spec = MPOSpec.full_rank((2, 2), (2, 2))
        self.weight = torch.randn(4, 4, dtype=torch.float64)
        self.inputs = torch.randn(3, 4, dtype=torch.float64)
        self.artifact = get_backend("native", "mpo").decompose(
            self.weight,
            self.spec,
        )

    def _assert_decomposition(self, provider: str) -> None:
        """检查一个计算库的原生分解和运行时。

        参数：
            provider: 本次检查选择的已注册 Provider 名称。
        """

        backend = get_backend(provider, "mpo")
        self.assertIsInstance(backend, TensorNetworkBackend)
        artifact = backend.decompose(self.weight, self.spec)
        reconstructed = reconstruct_mpo(artifact)
        torch.testing.assert_close(reconstructed, self.weight, rtol=1e-6, atol=1e-8)

        linear = backend.build_linear(artifact, trainable=False)
        self.assertIsInstance(linear, TensorNetworkLinear)
        exported = linear.export_artifact()
        torch.testing.assert_close(
            reconstruct_mpo(exported),
            reconstructed,
            rtol=1e-6,
            atol=1e-8,
        )
        expected = functional.linear(self.inputs, reconstructed)
        actual = linear(self.inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)

    def _assert_training(self, provider: str) -> None:
        """检查一个运行时的梯度和优化器更新。

        参数：
            provider: 本次检查选择的已注册 Provider 名称。
        """

        backend = get_backend(provider, "mpo")
        linear = backend.build_linear(self.artifact, trainable=True)
        before = linear.core_tensors()[0].detach().clone()
        optimizer = torch.optim.SGD(linear.parameters(), lr=1e-2)

        optimizer.zero_grad(set_to_none=True)
        linear(self.inputs).square().mean().backward()
        self.assertTrue(all(core.grad is not None for core in linear.core_tensors()))
        optimizer.step()

        self.assertFalse(torch.equal(before, linear.core_tensors()[0]))

    def test_registry_and_capabilities(self) -> None:
        """确认四个 Provider 声明的实际操作能力。"""

        self.assertEqual(
            list_backends("mpo"),
            ("native", "tensorly", "torchtt", "cutensornet"),
        )
        native = get_backend("native", "mpo")
        cutensornet = get_backend("cutensornet", "mpo")
        self.assertEqual(native.provider, "native")
        self.assertEqual(native.representation, "mpo")
        self.assertTrue(native.capabilities.training)
        self.assertFalse(cutensornet.capabilities.decomposition)
        self.assertFalse(cutensornet.capabilities.training)
        self.assertEqual(list_backends("tucker"), ())
        with self.assertRaises(ValueError):
            get_backend("tensorly", "tucker")
        with self.assertRaises(ValueError):
            get_backend("unknown", "mpo")

    def test_optional_backends_are_lazy(self) -> None:
        """确认导入 qcomp 时不会导入可选库。"""

        code = (
            "import sys, qcomp; "
            "assert 'tensorly' not in sys.modules; "
            "assert 'tltorch' not in sys.modules; "
            "assert 'torchtt' not in sys.modules; "
            "assert 'cuquantum' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_native_decomposition_and_training(self) -> None:
        """执行 native 分解、推理、梯度和参数更新。"""

        self._assert_decomposition("native")
        self._assert_training("native")

    @unittest.skipUnless(
        importlib.util.find_spec("tensorly")
        and importlib.util.find_spec("tltorch"),
        "TensorLy dependencies are not installed",
    )
    def test_tensorly_decomposition_and_training(self) -> None:
        """执行 TensorLy 分解、推理、梯度和参数更新。"""

        self._assert_decomposition("tensorly")
        self._assert_training("tensorly")

    @unittest.skipUnless(
        importlib.util.find_spec("torchtt"),
        "torchTT is not installed",
    )
    def test_torchtt_decomposition_and_training(self) -> None:
        """执行 torchTT 分解、推理、梯度和参数更新。"""

        self._assert_decomposition("torchtt")
        self._assert_training("torchtt")

    @unittest.skipUnless(
        importlib.util.find_spec("cuquantum")
        and torch.cuda.is_available(),
        "cuTensorNet with CUDA is not available",
    )
    def test_cutensornet_inference(self) -> None:
        """确认真实 cuTensorNet 收缩与稠密输出一致。"""

        backend = get_backend("cutensornet", "mpo")
        artifact = self.artifact.to("cuda", torch.float32)
        inputs = self.inputs.to("cuda", torch.float32)
        linear = backend.build_linear(artifact, trainable=False)
        self.assertIsInstance(linear, TensorNetworkLinear)
        torch.testing.assert_close(
            reconstruct_mpo(linear.export_artifact()),
            reconstruct_mpo(artifact),
        )

        with torch.inference_mode():
            actual = linear(inputs)
            expected = functional.linear(inputs, reconstruct_mpo(artifact))

        relative_error = (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        )
        self.assertLessEqual(float(relative_error), 5e-3)
        linear.close()

    def test_cutensornet_rejects_unsupported_operations(self) -> None:
        """确认 cuTensorNet 明确拒绝不支持的分解和训练操作。"""

        backend = get_backend("cutensornet", "mpo")
        with self.assertRaises(NotImplementedError):
            backend.decompose(self.weight, self.spec)
        with self.assertRaises(ValueError):
            backend.build_linear(self.artifact, trainable=True)


if __name__ == "__main__":
    unittest.main()
