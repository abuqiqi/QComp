"""验证常用评测指标的统一优化方向。

本模块检查 accuracy、exact match、perplexity 等指标方向、名称规范化和无效输入；同时
补充 NMSE 的数值正确性与边界校验。

主要内容：
- ``MetricDirectionTests``：验证单个与批量指标方向查询。
- ``NMSETests``：验证 NMSE 的标量计算与异常条件。
"""

from __future__ import annotations

import unittest

import torch

from qcomp import compute_nmse, metric_direction, resolve_metric_directions


class MetricDirectionTests(unittest.TestCase):
    """验证公共指标 registry 的查询和错误边界。"""

    def test_common_metrics_have_expected_directions(self) -> None:
        """常用 accuracy、exact match 和语言模型指标方向正确。"""

        for metric in (
            "acc",
            "acc_norm",
            "exact_match",
            "exact_match_remove_whitespace",
            "exact_match_strict_match",
            "exact_match_flexible_extract",
            "f1",
        ):
            with self.subTest(metric=metric):
                self.assertEqual(metric_direction(metric), "higher")
        for metric in (
            "loss",
            "perplexity",
            "word_perplexity",
            "byte_perplexity",
            "bits_per_byte",
        ):
            with self.subTest(metric=metric):
                self.assertEqual(metric_direction(metric), "lower")

    def test_resolve_normalizes_names_and_preserves_order(self) -> None:
        """批量解析会规范名称并保持调用方指定的指标顺序。"""

        directions = resolve_metric_directions((" ACC ", "Perplexity"))

        self.assertEqual(
            directions,
            {"acc": "higher", "perplexity": "lower"},
        )

    def test_invalid_metric_requests_are_rejected(self) -> None:
        """拒绝空、重复、未知指标以及误传的单个字符串。"""

        with self.assertRaises(TypeError):
            resolve_metric_directions("acc")
        for metrics in ((), (" ",), ("acc", " ACC "), ("unknown",)):
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    resolve_metric_directions(metrics)


class NMSETests(unittest.TestCase):
    """验证重构误差 NMSE 计算。"""

    def test_compute_nmse_matches_direct_formula(self) -> None:
        """检查 NMSE、MSE、目标能量与手工公式一致。"""

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 0.0, 2.0])

        result = compute_nmse(predictions=predictions, targets=targets)
        mse = torch.mean((predictions - targets) ** 2).item()
        target_power = torch.mean(targets**2).item()

        self.assertAlmostEqual(result["mse"].item(), mse)
        self.assertAlmostEqual(result["target_power"].item(), target_power)
        self.assertAlmostEqual(result["nmse"].item(), mse / target_power)

    def test_compute_nmse_supports_broadcast_and_rejects_invalid_shapes(self) -> None:
        """支持可广播的形状；不可广播时抛值错误。"""

        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([1.0, 2.0])
        self.assertIsInstance(compute_nmse(predictions=predictions, targets=targets), dict)
        with self.assertRaises(ValueError):
            compute_nmse(predictions=torch.randn(2, 3), targets=torch.randn(3, 1))

    def test_compute_nmse_rejects_small_target_power(self) -> None:
        """当 target 的平均能量过小会触发异常。"""

        predictions = torch.zeros(3)
        targets = torch.zeros(3)
        with self.assertRaises(ValueError):
            compute_nmse(predictions=predictions, targets=targets, eps=1e-6)


if __name__ == "__main__":
    unittest.main()
