"""验证与表示和后端无关的压缩指标及同步计时器。

本模块使用 MPO 和简单测试对象检查通用压缩指标、泛型分解配置、运行资源释放，以及
分解、推理和完整训练 step 计时。测试按照后端声明的能力执行可用操作。

主要内容：
- ``EvaluationTests``：测试压缩指标以及三类性能计时接口。
"""

import unittest
from unittest.mock import Mock, patch

import torch

from qcomp import MPOSpec, get_backend, list_backends
from qcomp.backends import BackendCapabilities
from qcomp.backends.native.mpo import NativeMPOLinear
from qcomp.evaluation import (
    compression_metrics,
    time_decomposition,
    time_inference,
    time_training_step,
)


class EvaluationTests(unittest.TestCase):
    """验证确定性的单层指标和计时结果。"""

    def setUp(self) -> None:
        """创建小型 native artifact 和固定训练数据。"""

        torch.manual_seed(11)
        self.backend = get_backend("native", "mpo")
        self.spec = MPOSpec.full_rank((2, 2), (2, 2))
        self.weight = torch.randn(4, 4, dtype=torch.float64)
        self.tn_artifact = self.backend.decompose(self.weight, self.spec)
        self.inputs = torch.randn(3, 4, dtype=torch.float64)
        self.targets = torch.randn(3, 4, dtype=torch.float64)

    def test_compression_metrics(self) -> None:
        """检查满 rank 精确重建、参数量和张量字节数统计。"""

        result = compression_metrics(self.weight, self.tn_artifact)

        self.assertEqual(result.dense_parameters, 16)
        self.assertEqual(result.compressed_parameters, 32)
        self.assertEqual(result.compression_ratio, 0.5)
        self.assertEqual(result.dense_tensor_bytes, 128)
        self.assertEqual(result.compressed_tensor_bytes, 256)
        self.assertEqual(result.tensor_size_compression_ratio, 0.5)
        self.assertLess(result.relative_error, 1e-12)

    def test_tensor_byte_metrics_respect_dtype(self) -> None:
        """使用张量数据类型计算实际张量字节数。"""

        float32_artifact = self.tn_artifact.to("cpu", torch.float32)
        result = compression_metrics(self.weight, float32_artifact)

        self.assertEqual(result.compression_ratio, 0.5)
        self.assertEqual(result.dense_tensor_bytes, 128)
        self.assertEqual(result.compressed_tensor_bytes, 128)
        self.assertEqual(result.tensor_size_compression_ratio, 1.0)

    def test_operation_timing(self) -> None:
        """测量分解、推理和完整训练 steps。"""

        decomposition = time_decomposition(
            self.backend,
            self.weight,
            self.spec,
            warmup=0,
            repeats=2,
        )
        inference = time_inference(
            self.backend,
            self.tn_artifact,
            self.inputs,
            warmup=0,
            repeats=2,
        )
        training = time_training_step(
            self.backend,
            self.tn_artifact,
            self.inputs,
            self.targets,
            warmup=0,
            repeats=2,
        )

        self.assertEqual(len(decomposition.samples_seconds), 2)
        self.assertGreater(inference.mean_seconds, 0.0)
        self.assertGreater(training.median_seconds, 0.0)

    def test_decomposition_timing_accepts_any_spec(self) -> None:
        """使用非 MPOSpec 对象执行通用分解计时。"""

        spec = object()
        backend = Mock()
        backend.provider = "example"
        backend.representation = "example"
        backend.capabilities = BackendCapabilities(
            decomposition=True,
            inference=False,
            training=False,
        )

        result = time_decomposition(
            backend,
            self.weight,
            spec,
            warmup=0,
            repeats=1,
        )

        self.assertGreater(result.mean_seconds, 0.0)
        backend.decompose.assert_called_once_with(self.weight, spec)

    def test_timing_closes_runtime_resources(self) -> None:
        """推理和训练计时正常结束后释放模型层资源。"""

        with patch.object(NativeMPOLinear, "close", autospec=True) as close:
            time_inference(
                self.backend,
                self.tn_artifact,
                self.inputs,
                warmup=0,
                repeats=1,
            )
            close.assert_called_once()

        with patch.object(NativeMPOLinear, "close", autospec=True) as close:
            time_training_step(
                self.backend,
                self.tn_artifact,
                self.inputs,
                self.targets,
                warmup=0,
                repeats=1,
            )
            close.assert_called_once()

    def test_timing_closes_runtime_after_failure(self) -> None:
        """推理和训练计时发生异常后仍释放模型层资源。"""

        with (
            patch.object(
                NativeMPOLinear,
                "forward",
                side_effect=RuntimeError("forward failed"),
            ),
            patch.object(NativeMPOLinear, "close", autospec=True) as close,
            self.assertRaisesRegex(RuntimeError, "forward failed"),
        ):
            time_inference(
                self.backend,
                self.tn_artifact,
                self.inputs,
                warmup=0,
                repeats=1,
            )
        close.assert_called_once()

        with (
            patch.object(
                NativeMPOLinear,
                "forward",
                side_effect=RuntimeError("forward failed"),
            ),
            patch.object(NativeMPOLinear, "close", autospec=True) as close,
            self.assertRaisesRegex(RuntimeError, "forward failed"),
        ):
            time_training_step(
                self.backend,
                self.tn_artifact,
                self.inputs,
                self.targets,
                warmup=0,
                repeats=1,
            )
        close.assert_called_once()

    def test_all_available_backends_can_be_timed(self) -> None:
        """执行每个可用后端支持的各项操作。"""

        for provider in list_backends("mpo"):
            backend = get_backend(provider, "mpo")
            if not backend.probe().available:
                continue
            if backend.capabilities.decomposition:
                result = time_decomposition(
                    backend,
                    self.weight,
                    self.spec,
                    warmup=0,
                    repeats=1,
                )
                self.assertGreater(result.mean_seconds, 0.0)

            device = torch.device(
                "cuda" if backend.capabilities.requires_cuda else "cpu"
            )
            inputs = self.inputs.to(device=device, dtype=torch.float32)
            inference = time_inference(
                backend,
                self.tn_artifact,
                inputs,
                warmup=1,
                repeats=1,
            )
            self.assertGreater(inference.mean_seconds, 0.0)

            if backend.capabilities.training:
                targets = self.targets.to(device=device, dtype=torch.float32)
                training = time_training_step(
                    backend,
                    self.tn_artifact,
                    inputs,
                    targets,
                    warmup=0,
                    repeats=1,
                )
                self.assertGreater(training.mean_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
