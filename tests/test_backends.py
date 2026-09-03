import importlib.util, os, subprocess, sys, unittest
from pathlib import Path

import torch
from qwen3_tn import (
    available_tt_backends,
    create_tt_linear,
    get_tt_backend,
    import_tt_backend_modules,
    probe_tt_backend,
    registered_tt_backends,
    register_tt_backend,
    tt_backend_metadata,
)
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
        code = (
            "import sys, qwen3_tn; "
            "assert 'tltorch' not in sys.modules; "
            "assert 'torchtt' not in sys.modules; "
            "assert 'cuquantum.tensornet' not in sys.modules"
        )
        environment = os.environ.copy()
        source_root = Path(__file__).resolve().parents[1] / "src"
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{python_path}" if python_path else str(source_root)
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_registered_available_and_capabilities_are_distinct(self):
        self.assertEqual(
            registered_tt_backends()[:4],
            ("native", "tensorly_torch", "torchtt", "cutensornet"),
        )
        self.assertIn("native", available_tt_backends())
        self.assertTrue(probe_tt_backend("native").available)
        cutensornet = get_tt_backend("cutensornet")
        self.assertFalse(cutensornet.capabilities.supports_training)
        self.assertFalse(cutensornet.capabilities.supports_backward)
        self.assertTrue(cutensornet.capabilities.requires_cuda)
        metadata = tt_backend_metadata("cutensornet", {"min_tokens": 4})
        self.assertEqual(metadata["options"]["min_tokens"], 4)
        self.assertEqual(metadata["options"]["token_bucket_size"], 128)

    def test_explicit_registration_and_options(self):
        import_tt_backend_modules(("dummy_backend",))
        self.assertIn("test_external", available_tt_backends())
        layer = create_tt_linear(
            self.spec,
            self.cores,
            tt_backend="test_external",
            trainable=True,
            backend_options={"marker": "received"},
        )
        output = layer(torch.randn(2, 4))
        output.sum().backward()
        self.assertTrue(all(core.grad is not None for core in layer.tt_parameters()))
        self.assertEqual(
            get_tt_backend("test_external").last_options,
            {"marker": "received"},
        )
        self.assertEqual(
            get_tt_backend("test_external").backend_metadata(),
            {"name": "test_external", "version": "test-1"},
        )

    def test_registration_errors_and_replace(self):
        from dummy_backend import DummyTTBackend

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_tt_backend("test_external", DummyTTBackend)
        register_tt_backend("test_external", DummyTTBackend, replace=True)
        with self.assertRaisesRegex(ValueError, "unknown TT backend"):
            get_tt_backend("absent_test_backend")
        with self.assertRaises(ModuleNotFoundError):
            import_tt_backend_modules(("qwen3_tn_absent_backend_module",))
        with self.assertRaisesRegex(ValueError, "does not accept options"):
            create_tt_linear(
                self.spec, self.cores, backend_options={"unsupported": True}
            )

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

    @unittest.skipUnless(importlib.util.find_spec("torchtt"), "torchTT not installed")
    def test_torchtt_matches_native_and_has_gradients(self):
        x = torch.randn(5, 4)
        native = create_tt_linear(self.spec, self.cores, trainable=True)
        torchtt = create_tt_linear(
            self.spec,
            self.cores,
            tt_backend="torchtt",
            trainable=True,
            token_chunk_size=2,
        )
        expected = native(x)
        actual = torchtt(x)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in torchtt.tt_parameters()
            )
        )
        saved = [
            parameter.detach().clone() for parameter in torchtt.tt_parameters()
        ]
        torchtt.load_cores([torch.zeros_like(core) for core in saved])
        torchtt.load_cores(saved)
        torch.testing.assert_close(torchtt(x), expected, atol=2e-5, rtol=2e-5)

    def test_cutensornet_rejects_training_and_invalid_options(self):
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            create_tt_linear(
                self.spec,
                self.cores,
                tt_backend="cutensornet",
                trainable=True,
            )
        with self.assertRaisesRegex(ValueError, "does not accept options"):
            create_tt_linear(
                self.spec,
                self.cores,
                tt_backend="cutensornet",
                backend_options={"unknown": True},
            )
        with self.assertRaisesRegex(ValueError, "min_tokens must be positive"):
            create_tt_linear(
                self.spec,
                self.cores,
                tt_backend="cutensornet",
                backend_options={"min_tokens": 0},
            )
        with self.assertRaisesRegex(RuntimeError, "activation checkpointing"):
            create_tt_linear(
                self.spec,
                self.cores,
                tt_backend="cutensornet",
                activation_checkpointing=True,
            )

    def test_cutensornet_uses_composition_for_native_fallback(self):
        from qwen3_tn.backends.cutensornet import CuTensorNetTTLinear
        from qwen3_tn.backends.native import NativeTTLinear

        self.assertFalse(issubclass(CuTensorNetTTLinear, NativeTTLinear))

    @unittest.skipUnless(
        importlib.util.find_spec("cuquantum") and torch.cuda.is_available(),
        "cuQuantum CUDA runtime not available",
    )
    def test_cutensornet_matches_native_inference(self):
        cores = [core.cuda().bfloat16() for core in self.cores]
        inputs = torch.randn(17, 4, device="cuda", dtype=torch.bfloat16)
        native = create_tt_linear(
            self.spec, cores, trainable=False, token_chunk_size=2
        ).cuda()
        cutensornet = create_tt_linear(
            self.spec,
            cores,
            tt_backend="cutensornet",
            trainable=False,
            backend_options={
                "min_tokens": 1,
                "token_bucket_size": 8,
                "max_cached_networks": 1,
                "memory_limit": "64MiB",
            },
        ).cuda()
        with torch.inference_mode():
            expected = native(inputs)
            actual = cutensornet(inputs)
        relative_l2 = torch.linalg.vector_norm(
            actual.float() - expected.float()
        ) / torch.linalg.vector_norm(expected.float())
        self.assertLessEqual(float(relative_l2), 5e-3)
        saved = [core.detach().clone() for core in cutensornet.tt_parameters()]
        cutensornet.load_cores([torch.zeros_like(core) for core in saved])
        cutensornet.load_cores(saved)
        with torch.inference_mode():
            restored = cutensornet(inputs)
        torch.testing.assert_close(restored, actual)
        cutensornet.cpu()
