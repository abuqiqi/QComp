"""验证面向任务的 qcomp 工作流。

本模块使用小型 PyTorch 模型检查单层与模型级压缩编排，确认 workflow 复用 backend
的分解和模型层构造能力，并验证批量压缩的输入检查、结果顺序与原子回滚。

主要内容：
- ``WorkflowTests``：验证单层压缩、批量压缩、失败回滚和原始层恢复。
- ``_FailingSecondDecompositionBackend``：模拟批量分解中途失败。
- ``_TrackingExecutionBackend``：记录回滚时是否释放已构造模型层。
"""

import unittest

import torch
from torch import nn

from qcomp import (
    CompressionPlan,
    CompressionTarget,
    MPOSpec,
    compress_linear,
    compress_model,
    get_backend,
    restore_compressed_model,
    restore_linear,
)
from qcomp.backends.native.mpo import NativeMPOBackend, NativeMPOLinear
from qcomp.nn import TensorNetworkLinear
from qcomp.representations import TensorNetworkArtifact


class _TrackingNativeMPOLinear(NativeMPOLinear):
    """记录模型级回滚是否释放已安装的 native 压缩层。"""

    def __init__(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> None:
        """构造可记录关闭状态的 native MPO Linear。

        参数：
            artifact: 用于构造模型层的规范 MPO artifact。
            trainable: cores 是否需要参与训练。
        """

        super().__init__(artifact, trainable=trainable)
        self.closed = False

    def close(self) -> None:
        """记录资源释放操作并调用父类关闭逻辑。"""

        self.closed = True
        super().close()


class _TrackingExecutionBackend(NativeMPOBackend):
    """构造能够记录关闭状态的 native MPO Linear。"""

    def __init__(self) -> None:
        """初始化本次测试创建的模型层列表。"""

        self.created: list[_TrackingNativeMPOLinear] = []

    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> _TrackingNativeMPOLinear:
        """构造并记录一个可跟踪关闭状态的模型层。

        参数：
            artifact: 用于构造模型层的规范 MPO artifact。
            trainable: cores 是否需要参与训练。

        返回：
            新建的可跟踪 native MPO Linear。
        """

        linear = _TrackingNativeMPOLinear(artifact, trainable=trainable)
        self.created.append(linear)
        return linear


class _FailingSecondDecompositionBackend(NativeMPOBackend):
    """在第二次分解时失败，用于验证模型级原子回滚。"""

    def __init__(self) -> None:
        """初始化分解调用次数。"""

        self.calls = 0

    def decompose(
        self,
        weight: torch.Tensor,
        spec: MPOSpec,
    ) -> TensorNetworkArtifact:
        """执行第一次 native 分解，并让第二次分解失败。

        参数：
            weight: 待分解的稠密权重。
            spec: 对应目标层的 MPO 配置。

        返回：
            第一次调用产生的规范 MPO artifact。

        异常：
            RuntimeError: 第二次调用时抛出。
        """

        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("planned decomposition failure")
        return super().decompose(weight, spec)


class _OtherRepresentationBackend(NativeMPOBackend):
    """声明测试表示，并记录混合计划是否选择对应 backend。"""

    representation = "tucker"

    def __init__(self) -> None:
        """初始化分解调用次数。"""

        self.decomposition_calls = 0

    def decompose(
        self,
        weight: torch.Tensor,
        spec: MPOSpec,
    ) -> TensorNetworkArtifact:
        """记录调用并复用 native MPO 生成测试 artifact。

        参数：
            weight: 待分解的稠密权重。
            spec: 测试使用的 MPO 配置。

        返回：
            用于验证工作流编排的 MPO artifact。
        """

        self.decomposition_calls += 1
        return super().decompose(weight, spec)


class WorkflowTests(unittest.TestCase):
    """验证单层压缩工作流的完整结果。"""

    def test_integer_decomposition_dtype_rejected_before_replacement(self) -> None:
        """拒绝整数分解精度并保持原始模型层。"""

        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        original = model[0]
        backend = get_backend("native", "mpo")
        with self.assertRaisesRegex(ValueError, "floating-point"):
            compress_linear(
                model,
                "0",
                MPOSpec.full_rank((2, 2), (2, 2)),
                decomposition_backend=backend,
                execution_backend=backend,
                decomposition_dtype=torch.int64,
                trainable=False,
            )
        self.assertIs(model[0], original)

    def test_compress_linear_and_restore(self) -> None:
        """压缩 Sequential 中的 Linear，并通过替换记录恢复原层。"""

        torch.manual_seed(13)
        model = nn.Sequential(nn.Linear(4, 4, bias=False)).to(torch.float64)
        original = model[0]
        inputs = torch.randn(3, 4, dtype=torch.float64)
        expected = model(inputs)
        backend = get_backend("native", "mpo")

        result = compress_linear(
            model,
            "0",
            MPOSpec.full_rank((2, 2), (2, 2)),
            decomposition_backend=backend,
            execution_backend=backend,
            trainable=False,
        )

        self.assertIsInstance(model[0], TensorNetworkLinear)
        self.assertEqual(result.tn_artifact.representation, "mpo")
        torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-8)

        restore_linear(model, result.replacement)
        self.assertIs(model[0], original)


    def test_compress_and_restore_model(self) -> None:
        """批量压缩多个 Linear，并按计划顺序返回和恢复各层。"""

        torch.manual_seed(29)
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.ReLU(),
            nn.Linear(4, 4, bias=False),
        ).to(torch.float64)
        originals = (model[0], model[2])
        inputs = torch.randn(3, 4, dtype=torch.float64)
        expected = model(inputs)
        spec = MPOSpec.full_rank((2, 2), (2, 2))
        plan = CompressionPlan(
            targets=(
                CompressionTarget("0", "MPO", spec),
                CompressionTarget("2", "mpo", spec),
            ),
        )
        backend = get_backend("native", "mpo")

        result = compress_model(
            model,
            plan,
            decomposition_backends={"mpo": backend},
            execution_backends={"mpo": backend},
            trainable=False,
        )

        self.assertEqual(plan.targets[0].representation, "mpo")
        self.assertEqual(
            tuple(item.replacement.target for item in result.layer_results),
            ("0", "2"),
        )
        self.assertTrue(
            all(
                item.tn_artifact.representation == "mpo"
                for item in result.layer_results
            )
        )
        self.assertIsInstance(model[0], TensorNetworkLinear)
        self.assertIsInstance(model[2], TensorNetworkLinear)
        torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-8)

        restore_compressed_model(model, result)
        self.assertIs(model[0], originals[0])
        self.assertIs(model[2], originals[1])

    def test_compression_plan_validation(self) -> None:
        """拒绝空计划、空目标路径和重复目标路径。"""

        spec = MPOSpec.full_rank((2, 2), (2, 2))
        with self.assertRaisesRegex(ValueError, "targets"):
            CompressionPlan(())
        with self.assertRaisesRegex(ValueError, "module_path"):
            CompressionTarget("", "mpo", spec)
        with self.assertRaisesRegex(ValueError, "unique"):
            CompressionPlan(
                (
                    CompressionTarget("0", "mpo", spec),
                    CompressionTarget("0", "mpo", spec),
                ),
            )

    def test_preflight_validation_does_not_modify_model(self) -> None:
        """在 backend 或目标路径检查失败时保持模型结构不变。"""

        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        original = model[0]
        spec = MPOSpec.full_rank((2, 2), (2, 2))
        backend = get_backend("native", "mpo")
        unknown_target_plan = CompressionPlan(
            (
                CompressionTarget("0", "mpo", spec),
                CompressionTarget("missing", "mpo", spec),
            ),
        )
        with self.assertRaises(AttributeError):
            compress_model(
                model,
                unknown_target_plan,
                decomposition_backends={"mpo": backend},
                execution_backends={"mpo": backend},
                trainable=False,
            )
        self.assertIs(model[0], original)

        valid_plan = CompressionPlan(
            (CompressionTarget("0", "mpo", spec),),
        )
        with self.assertRaisesRegex(ValueError, "declares representation"):
            compress_model(
                model,
                valid_plan,
                decomposition_backends={"mpo": backend},
                execution_backends={"mpo": _OtherRepresentationBackend()},
                trainable=False,
            )
        self.assertIs(model[0], original)

    def test_compress_model_selects_backends_for_mixed_representations(self) -> None:
        """按照每个目标声明的表示选择外部 backend。"""

        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        ).to(torch.float64)
        originals = (model[0], model[1])
        spec = MPOSpec.full_rank((2, 2), (2, 2))
        plan = CompressionPlan(
            (
                CompressionTarget("0", "mpo", spec),
                CompressionTarget("1", "TUCKER", spec),
            ),
        )
        mpo_backend = get_backend("native", "mpo")
        tucker_backend = _OtherRepresentationBackend()

        result = compress_model(
            model,
            plan,
            decomposition_backends={
                "mpo": mpo_backend,
                "tucker": tucker_backend,
            },
            execution_backends={
                "mpo": mpo_backend,
                "tucker": tucker_backend,
            },
            trainable=False,
        )

        self.assertEqual(tucker_backend.decomposition_calls, 1)
        self.assertEqual(plan.targets[1].representation, "tucker")
        restore_compressed_model(model, result)
        self.assertIs(model[0], originals[0])
        self.assertIs(model[1], originals[1])

    def test_missing_backend_does_not_modify_model(self) -> None:
        """缺少目标表示的 backend 时在替换任何层前失败。"""

        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        original = model[0]
        plan = CompressionPlan(
            (
                CompressionTarget(
                    "0",
                    "mpo",
                    MPOSpec.full_rank((2, 2), (2, 2)),
                ),
            ),
        )
        backend = get_backend("native", "mpo")

        with self.assertRaisesRegex(ValueError, "missing execution backend"):
            compress_model(
                model,
                plan,
                decomposition_backends={"mpo": backend},
                execution_backends={},
                trainable=False,
            )

        self.assertIs(model[0], original)

    def test_compress_model_rolls_back_after_decomposition_failure(self) -> None:
        """中间分解失败时恢复并关闭此前安装的压缩层。"""

        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        ).to(torch.float64)
        originals = (model[0], model[1])
        spec = MPOSpec.full_rank((2, 2), (2, 2))
        plan = CompressionPlan(
            (
                CompressionTarget("0", "mpo", spec),
                CompressionTarget("1", "mpo", spec),
            ),
        )
        decomposition_backend = _FailingSecondDecompositionBackend()
        execution_backend = _TrackingExecutionBackend()

        with self.assertRaisesRegex(RuntimeError, "planned"):
            compress_model(
                model,
                plan,
                decomposition_backends={"mpo": decomposition_backend},
                execution_backends={"mpo": execution_backend},
                trainable=False,
            )

        self.assertIs(model[0], originals[0])
        self.assertIs(model[1], originals[1])
        self.assertEqual(len(execution_backend.created), 1)
        self.assertTrue(execution_backend.created[0].closed)


if __name__ == "__main__":
    unittest.main()
