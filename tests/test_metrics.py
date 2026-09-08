"""验证常用评测指标的统一优化方向。

本模块检查 accuracy、exact match、perplexity 等指标的方向解析、名称规范化和无效输入，
不运行模型或真实 benchmark。

主要内容：
- ``MetricDirectionTests``：验证单个与批量指标方向查询。
"""

from __future__ import annotations

import unittest

from qcomp import metric_direction, resolve_metric_directions


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


if __name__ == "__main__":
    unittest.main()
