"""验证 Qwen3 矩阵 NMSE 脚本的辅助输出。

本模块不执行模型加载，仅覆盖路径解析、CSV 输出与默认文件名映射，保证默认导出行为稳定。
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import run_qwen3_matrix_nmse as matrix_nmse


class TestMatrixNMScript(unittest.TestCase):
    """验证 matrix nmse 脚本的基础工具函数。"""

    def test_extract_module_coordinates(self) -> None:
        """标准层路径可提取 block 与模块名。"""

        block, module = matrix_nmse._extract_module_coordinates(
            "model.layers.18.self_attn.q_proj"
        )
        self.assertEqual((block, module), (18, "q_proj"))

    def test_invalid_module_coordinates(self) -> None:
        """不匹配 Qwen3 层路径时抛出异常。"""

        with self.assertRaises(ValueError):
            matrix_nmse._extract_module_coordinates("transformer.blocks.1.dense")

    def test_default_output_paths(self) -> None:
        """默认输出路径会推断为 json/csv/heatmap 三类文件。"""

        json_path, csv_path, heatmap_path = matrix_nmse._model_output_paths(
            "/tmp/run/sample"
        )
        self.assertEqual(json_path.name, "sample.json")
        self.assertEqual(csv_path.name, "sample.csv")
        self.assertEqual(heatmap_path.name, "sample-nmse-heatmap.png")

    def test_write_case_csv(self) -> None:
        """明细 CSV 保留字段顺序并能写入字符串/数值。"""

        records = [
            {
                "module": "model.layers.0.self_attn.q_proj",
                "status": "ok",
                "module_rank": 96,
                "out_features": 4,
                "in_features": 8,
                "numel": 32,
                "nmse": 0.12,
                "mse": 0.3,
                "target_power": 2.5,
            }
        ]

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "detail.csv"
            matrix_nmse.write_case_csv(records, output)
            rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(
            rows[0],
            [
                "module",
                "status",
                "module_rank",
                "out_features",
                "in_features",
                "numel",
                "nmse",
                "mse",
                "target_power",
                "error",
            ],
        )
        self.assertEqual(rows[1][0], "model.layers.0.self_attn.q_proj")
        self.assertEqual(rows[1][1], "ok")
