"""验证局部输出 NMSE 的 mask、跨 batch 归约和输入校验。"""

import pytest
import torch

from qcomp.evaluation import local_output_error_sums, local_output_nmse


def test_mask_excludes_padding_and_aggregates_batches() -> None:
    """padding 误差不计入，多个 batch 必须先合并分子和分母。"""
    reference_a = torch.tensor([[[1.0, 2.0], [10.0, 10.0]]])
    compressed_a = torch.tensor([[[2.0, 2.0], [99.0, 99.0]]])
    error_a, power_a, tokens_a = local_output_error_sums(
        reference_a, compressed_a, torch.tensor([[1, 0]])
    )
    reference_b = torch.tensor([[[2.0, 0.0]]])
    compressed_b = torch.tensor([[[0.0, 0.0]]])
    error_b, power_b, tokens_b = local_output_error_sums(
        reference_b, compressed_b, torch.tensor([[1]])
    )
    assert tokens_a + tokens_b == 2
    assert local_output_nmse(error_a + error_b, power_a + power_b) == pytest.approx(
        5 / 9
    )


def test_identical_and_zero_reference_are_finite() -> None:
    """相同输出为零误差，零能量 reference 仍产生有限结果。"""
    value = torch.zeros(1, 2, 3)
    error, power, _ = local_output_error_sums(value, value, torch.ones(1, 2))
    assert local_output_nmse(error, power) == 0.0
    assert local_output_nmse(1.0, 0.0) == pytest.approx(1e12)


@pytest.mark.parametrize(
    ("reference", "compressed", "mask"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), torch.ones(1, 2)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3)),
    ],
)
def test_error_sums_reject_invalid_shapes(reference, compressed, mask) -> None:
    """输出不是三维、输出不对齐或 mask 不对齐时明确失败。"""
    with pytest.raises(ValueError):
        local_output_error_sums(reference, compressed, mask)


@pytest.mark.parametrize(
    ("error", "power", "epsilon"),
    [(-1.0, 1.0, 1e-12), (1.0, -1.0, 1e-12), (1.0, 1.0, 0.0)],
)
def test_nmse_rejects_invalid_sums(error, power, epsilon) -> None:
    """累计量必须非负且稳定项必须为正。"""
    with pytest.raises(ValueError):
        local_output_nmse(error, power, epsilon=epsilon)
