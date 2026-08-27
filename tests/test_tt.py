import unittest
import torch
from qwen3_tn.tt import (
    TTMatrixSpec,
    detensorize_matrix,
    reconstruct_matrix,
    slice_bond,
    tensorize_matrix,
    tt_svd_matrix,
)


class TTTests(unittest.TestCase):
    def test_full_rank_round_trip(self):
        spec = TTMatrixSpec.full_rank((2, 2), (2, 2))
        weight = torch.randn(4, 4)
        cores = tt_svd_matrix(weight, spec, svd_driver=None)
        torch.testing.assert_close(
            reconstruct_matrix(cores, spec), weight, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            detensorize_matrix(tensorize_matrix(weight, spec), spec), weight
        )

    def test_validation_and_slice(self):
        spec = TTMatrixSpec((2, 2), (2, 2), (1, 4, 1))
        cores = tt_svd_matrix(torch.randn(4, 4), spec, svd_driver=None)
        sliced, smaller = slice_bond(cores, spec, bond_index=1, rank=2)
        self.assertEqual(smaller.ranks, (1, 2, 1))
        self.assertEqual(sliced[0].shape[-1], 2)

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            TTMatrixSpec((2,), (2,), (2, 2))
