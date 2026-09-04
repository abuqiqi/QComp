"""验证完整 Causal LM 正常生成和运行开销统计。

本模块使用具有确定性 generate 方法的小型模型和 DataLoader，检查生成配置、CPU token
结果、输入与生成 token 计数、时间、显存字段和模型 train/eval 状态恢复。

主要内容：
- ``_TinyGenerativeModel``：记录生成配置并返回固定长度的新 token。
- ``_SamplingGenerativeModel``：使用 PyTorch 随机数验证采样 seed。
- ``InferenceWorkflowTests``：验证正常推理结果和输入错误。
"""

import unittest
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from qcomp import InferenceConfig, infer_causal_lm


class _TinyGenerativeModel(nn.Module):
    """记录运行状态并追加固定的新 token。"""

    def __init__(self) -> None:
        """创建用于支持设备迁移的标量 buffer 和调用记录。"""

        super().__init__()
        self.register_buffer("marker", torch.tensor(0))
        self.calls: list[tuple[bool, bool, int, bool]] = []

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Tensor:
        """为每条输入追加指定数量的固定 token。

        参数：
            input_ids: 二维输入 token IDs。
            attention_mask: 可选的输入有效位置掩码。
            max_new_tokens: 需要追加的新 token 数。
            do_sample: 是否使用采样生成。

        返回：
            包含输入和新增 token 的完整序列。
        """

        del attention_mask
        self.calls.append(
            (
                self.training,
                torch.is_grad_enabled(),
                max_new_tokens,
                do_sample,
            )
        )
        new_tokens = torch.full(
            (input_ids.shape[0], max_new_tokens),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, new_tokens), dim=-1)


class _SamplingGenerativeModel(nn.Module):
    """使用全局 PyTorch 随机数生成测试 token。"""

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Tensor:
        """根据推理配置追加随机或固定 token。

        参数：
            input_ids: 二维输入 token IDs。
            attention_mask: 可选的输入有效位置掩码。
            max_new_tokens: 需要追加的新 token 数。
            do_sample: 是否使用随机采样。

        返回：
            包含输入和新增 token 的完整序列。
        """

        del attention_mask
        if do_sample:
            new_tokens = torch.randint(
                0,
                16,
                (input_ids.shape[0], max_new_tokens),
                device=input_ids.device,
            )
        else:
            new_tokens = torch.zeros(
                (input_ids.shape[0], max_new_tokens),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        return torch.cat((input_ids, new_tokens), dim=-1)


def _make_inference_loader() -> DataLoader[Any]:
    """创建两个包含 attention mask 和 labels 的确定性 batch。"""

    examples = (
        {
            "input_ids": torch.tensor([1, 2, 0]),
            "attention_mask": torch.tensor([1, 1, 0]),
            "labels": torch.tensor([1, 2, -100]),
        },
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([3, 4, 5]),
        },
    )
    return DataLoader(examples, batch_size=1, shuffle=False)


class InferenceWorkflowTests(unittest.TestCase):
    """验证正常生成 workflow 的输出和性能数据。"""

    def test_infer_causal_lm_returns_tokens_and_performance(self) -> None:
        """返回 CPU 新 token，并统计实际时间、吞吐和输入规模。"""

        model = _TinyGenerativeModel()
        model.train()
        result = infer_causal_lm(
            model,
            _make_inference_loader(),
            InferenceConfig(
                device="cpu",
                max_new_tokens=2,
                do_sample=False,
                seed=11,
            ),
        )

        self.assertEqual(len(result.generated_token_ids), 2)
        for generated in result.generated_token_ids:
            self.assertEqual(generated.device.type, "cpu")
            torch.testing.assert_close(generated, torch.tensor([[7, 7]]))
        self.assertEqual(
            model.calls,
            [(False, False, 2, False), (False, False, 2, False)],
        )
        self.assertTrue(model.training)
        self.assertEqual(result.performance.input_tokens, 5)
        self.assertEqual(result.performance.generated_tokens, 4)
        self.assertEqual(len(result.performance.batch_seconds), 2)
        self.assertGreater(result.performance.generation_seconds, 0)
        self.assertGreaterEqual(
            result.performance.total_seconds,
            result.performance.generation_seconds,
        )
        self.assertGreater(result.performance.generated_tokens_per_second, 0)
        self.assertIsNone(result.performance.peak_allocated_bytes)
        self.assertIsNone(result.performance.peak_reserved_bytes)

    def test_sampling_uses_configured_seed(self) -> None:
        """使用相同 seed 的采样推理返回相同 token。"""

        config = InferenceConfig(
            device="cpu",
            max_new_tokens=4,
            do_sample=True,
            seed=23,
        )
        first = infer_causal_lm(
            _SamplingGenerativeModel(),
            _make_inference_loader(),
            config,
        )
        second = infer_causal_lm(
            _SamplingGenerativeModel(),
            _make_inference_loader(),
            config,
        )

        for actual, expected in zip(
            first.generated_token_ids,
            second.generated_token_ids,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)

    def test_inference_validates_config_and_batches(self) -> None:
        """拒绝无效生成长度、非映射 batch 和缺少 input_ids 的 batch。"""

        with self.assertRaisesRegex(ValueError, "positive"):
            InferenceConfig(max_new_tokens=0)

        tensor_loader = DataLoader((torch.tensor([1, 2]),), batch_size=1)
        with self.assertRaisesRegex(ValueError, "mappings"):
            infer_causal_lm(
                _TinyGenerativeModel(),
                tensor_loader,
                InferenceConfig(device="cpu"),
            )

        model = _TinyGenerativeModel()
        model.train()
        missing_input_loader = DataLoader(
            ({"attention_mask": torch.tensor([1, 1])},),
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "input_ids"):
            infer_causal_lm(
                model,
                missing_input_loader,
                InferenceConfig(device="cpu"),
            )
        self.assertTrue(model.training)

    def test_unavailable_cuda_is_rejected(self) -> None:
        """请求不可用 CUDA 时在移动模型前抛出错误。"""

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                infer_causal_lm(
                    _TinyGenerativeModel(),
                    _make_inference_loader(),
                    InferenceConfig(device="cuda:0"),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_peak_memory_is_reported(self) -> None:
        """CUDA 推理返回 PyTorch allocator 的峰值显存。"""

        result = infer_causal_lm(
            _TinyGenerativeModel(),
            _make_inference_loader(),
            InferenceConfig(device="cuda:0", max_new_tokens=2),
        )

        self.assertIsInstance(result.performance.peak_allocated_bytes, int)
        self.assertIsInstance(result.performance.peak_reserved_bytes, int)
        self.assertGreaterEqual(result.performance.peak_allocated_bytes, 0)
        self.assertGreaterEqual(result.performance.peak_reserved_bytes, 0)
