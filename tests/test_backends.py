import importlib.util, sys, unittest
import torch
from qwen3_tn import create_tt_linear
from qwen3_tn.tt import TTMatrixSpec, reconstruct_matrix, tt_svd_matrix


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.spec = TTMatrixSpec((2, 2), (2, 2), (1, 2, 1))
        self.cores = tt_svd_matrix(torch.randn(4, 4), self.spec, svd_driver=None)

    def test_native_forward_and_grad(self):
        layer = create_tt_linear(
            self.spec,
            self.cores,
            trainable=True,
            token_chunk_size=2,
            activation_checkpointing=True,
        )
        inputs = torch.randn(2, 3, 4, requires_grad=True)
        output = layer(inputs)
        expected = inputs @ reconstruct_matrix(self.cores, self.spec).T
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        output.square().mean().backward()
        self.assertTrue(
            all(
                core.grad is not None and torch.isfinite(core.grad).all()
                for core in layer.tt_parameters()
            )
        )

    def test_optional_backend_is_lazy(self):
        self.assertNotIn("tltorch", sys.modules)

    @unittest.skipUnless(
        importlib.util.find_spec("tltorch"), "TensorLy-Torch not installed"
    )
    def test_tensorly_matches_native(self):
        x = torch.randn(5, 4)
        native = create_tt_linear(self.spec, self.cores, trainable=True)
        tensorly = create_tt_linear(
            self.spec, self.cores, tt_backend="tensorly_torch", trainable=True
        )
        torch.testing.assert_close(native(x), tensorly(x), atol=2e-5, rtol=2e-5)
