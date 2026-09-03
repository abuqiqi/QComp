"""验证规范 MPO 结构配置、artifact 和矩阵变换。

本模块使用不依赖外部后端的小型确定性张量，检查张量化闭环、满 rank core 重建、
artifact 解析以及稠密参数量与压缩参数量统计。

主要内容：
- ``MPOTests``：测试 MPOSpec、规范 cores、artifact 转换和稠密权重重建。
"""

import unittest

import torch

from qcomp import reconstruct_tensor
from qcomp.representations.mpo import (
    MPOSpec,
    detensorize_matrix,
    make_mpo_artifact,
    parse_mpo_artifact,
    reconstruct_mpo,
    tensorize_matrix,
)


class MPOTests(unittest.TestCase):
    """验证规范布局、维度和稠密权重重建。"""

    def test_tensorize_round_trip(self) -> None:
        """确认稠密矩阵经过成对物理 modes 转换后保持不变。"""

        spec = MPOSpec.full_rank((2, 2), (2, 2))
        weight = torch.arange(16, dtype=torch.float64).reshape(4, 4)

        tensorized = tensorize_matrix(weight, spec)
        reconstructed = detensorize_matrix(tensorized, spec)

        torch.testing.assert_close(reconstructed, weight)

    def test_artifact_reconstructs_canonical_cores(self) -> None:
        """确认 artifact 重建结果与直接 core 收缩结果一致。"""

        spec = MPOSpec((2, 2), (2, 2), (1, 1, 1))
        first = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2, 1)
        second = torch.arange(4, 8, dtype=torch.float64).reshape(1, 2, 2, 1)
        artifact = make_mpo_artifact(spec, (first, second))

        parsed_spec, parsed_cores = parse_mpo_artifact(artifact)

        self.assertEqual(parsed_spec, spec)
        self.assertEqual(tuple(core.shape for core in parsed_cores), spec.core_shapes)
        self.assertEqual(tuple(reconstruct_mpo(artifact).shape), (4, 4))

    def test_parameter_counts(self) -> None:
        """根据一个结构配置统计稠密与压缩参数量。"""

        spec = MPOSpec((2, 2), (2, 2), (1, 1, 1))

        self.assertEqual(spec.dense_num_parameters, 16)
        self.assertEqual(spec.num_parameters, 8)
        self.assertEqual(spec.compression_ratio, 2.0)

    def test_generic_reconstruction_dispatch(self) -> None:
        """根据 artifact 表示名称自动调用 MPO 重建函数。"""

        spec = MPOSpec((2, 2), (2, 2), (1, 1, 1))
        first = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2, 1)
        second = torch.arange(4, 8, dtype=torch.float64).reshape(1, 2, 2, 1)
        tn_artifact = make_mpo_artifact(spec, (first, second))

        torch.testing.assert_close(
            reconstruct_tensor(tn_artifact),
            reconstruct_mpo(tn_artifact),
        )


if __name__ == "__main__":
    unittest.main()
