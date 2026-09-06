"""验证通用 artifact 的张量转换和持久化行为。

本模块覆盖命名张量的数据类型转换与保存加载闭环，确保表示类型、元数据、张量名称和
数值均被保留。测试不依赖 MPO 或任何具体计算后端。

主要内容：
- ``ArtifactTests``：测试 artifact 转换、序列化和反序列化。
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from qcomp import (
    ArtifactPaths,
    TensorNetworkArtifact,
    load_artifact,
    reconstruct_tensor,
    save_artifact,
)


class ArtifactTests(unittest.TestCase):
    """验证通用 artifact 不依赖具体张量网络表示。"""

    def test_save_and_load_generic_artifact(self) -> None:
        """在磁盘保存过程中保留表示元数据和命名张量。"""

        artifact = TensorNetworkArtifact(
            representation="example",
            metadata={"shape": [2, 2]},
            tensors={"factor": torch.arange(4).reshape(2, 2).float()},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            save_artifact(path, artifact)
            loaded = load_artifact(path)

        self.assertEqual(loaded.representation, "example")
        self.assertEqual(loaded.metadata, {"shape": [2, 2]})
        torch.testing.assert_close(loaded.tensors["factor"], artifact.tensors["factor"])

    def test_artifact_paths_create_standard_directories(self) -> None:
        """在自定义根目录下创建全部标准实验产物目录。"""

        default_paths = ArtifactPaths()
        self.assertEqual(default_paths.root, Path("artifacts"))
        self.assertEqual(default_paths.evaluations, Path("artifacts/evaluations"))

        with TemporaryDirectory() as directory:
            paths = ArtifactPaths(Path(directory) / "outputs")
            paths.create_directories()

            self.assertEqual(paths.datasets, paths.root / "datasets")
            self.assertEqual(paths.cache, paths.root / "cache")
            self.assertEqual(paths.decompositions, paths.root / "decompositions")
            self.assertEqual(paths.checkpoints, paths.root / "checkpoints")
            self.assertEqual(paths.evaluations, paths.root / "evaluations")
            for path in (
                paths.datasets,
                paths.cache,
                paths.decompositions,
                paths.checkpoints,
                paths.evaluations,
            ):
                self.assertTrue(path.is_dir())

    def test_move_all_artifact_tensors(self) -> None:
        """转换全部命名张量，同时保持结构元数据不变。"""

        artifact = TensorNetworkArtifact(
            representation="example",
            metadata={"rank": 2},
            tensors={"left": torch.ones(2), "right": torch.zeros(2)},
        )
        converted = artifact.to("cpu", torch.float64)

        self.assertEqual(converted.metadata, artifact.metadata)
        self.assertTrue(
            all(tensor.dtype == torch.float64 for tensor in converted.tensors.values())
        )

    def test_unknown_representation_cannot_be_reconstructed(self) -> None:
        """拒绝没有注册稠密重建函数的表示类型。"""

        tn_artifact = TensorNetworkArtifact(
            representation="unknown",
            metadata={},
            tensors={"factor": torch.ones(1)},
        )

        with self.assertRaisesRegex(
            ValueError,
            "no reconstructor registered for representation: 'unknown'",
        ):
            reconstruct_tensor(tn_artifact)


if __name__ == "__main__":
    unittest.main()
