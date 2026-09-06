"""验证统一数据来源配置与懒加载行为。

本模块使用临时路径和模拟的 Hugging Face Datasets 模块，检查 Hub、JSONL 与本地磁盘
Dataset 的参数分派、split 选择和路径校验。独立子进程确认导入 qcomp 不会加载可选的
``datasets`` 依赖。

主要内容：
- ``DataSourceTests``：验证三种来源配置、统一加载入口和依赖懒加载。
- ``_FakeDatasetDict``：模拟可按 split 选择的 Hugging Face DatasetDict。
"""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import ANY, Mock, patch

from qcomp import (
    HuggingFaceDatasetSource,
    JsonlDatasetSource,
    LocalDatasetSource,
    load_dataset_source,
)


class _FakeDatasetDict(dict[str, object]):
    """模拟 Hugging Face DatasetDict 的 split 映射行为。"""


class _FakeDownloadConfig:
    """记录 datasets 下载配置中的本地文件开关。"""

    def __init__(self, *, local_files_only: bool) -> None:
        """保存是否只允许读取本地文件。"""

        self.local_files_only = local_files_only


def _datasets_module(
    *,
    loaded_dataset: object,
) -> tuple[ModuleType, Mock, Mock]:
    """构造只包含统一加载入口所需对象的模拟 datasets 模块。

    参数：
        loaded_dataset: ``load_from_disk`` 应返回的测试对象。

    返回：
        模拟模块、load_dataset mock 和 load_from_disk mock。
    """

    module = ModuleType("datasets")
    load_dataset = Mock(return_value=object())
    load_from_disk = Mock(return_value=loaded_dataset)
    module.DatasetDict = _FakeDatasetDict
    module.DownloadConfig = _FakeDownloadConfig
    module.load_dataset = load_dataset
    module.load_from_disk = load_from_disk
    return module, load_dataset, load_from_disk


class DataSourceTests(unittest.TestCase):
    """验证不同数据位置通过同一个加载函数返回 Dataset。"""

    def test_huggingface_source_forwards_dataset_configuration(self) -> None:
        """向 load_dataset 传递 Hub 名称、子配置、split 和缓存路径。"""

        module, load_dataset, _ = _datasets_module(loaded_dataset=object())
        source = HuggingFaceDatasetSource(
            dataset=" cais/mmlu ",
            config="abstract_algebra",
            split=" test ",
            cache_dir="cache",
        )
        with patch.dict(sys.modules, {"datasets": module}):
            loaded = load_dataset_source(source)

        self.assertIs(loaded, load_dataset.return_value)
        load_dataset.assert_called_once_with(
            "cais/mmlu",
            "abstract_algebra",
            split="test",
            cache_dir="cache",
            download_config=ANY,
        )
        self.assertTrue(
            load_dataset.call_args.kwargs["download_config"].local_files_only
        )

    def test_jsonl_source_uses_named_local_split(self) -> None:
        """把本地 JSONL 文件作为指定逻辑 split 交给 load_dataset。"""

        module, load_dataset, _ = _datasets_module(loaded_dataset=object())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"text": "example"}\n', encoding="utf-8")
            source = JsonlDatasetSource(path, split="validation", cache_dir="cache")
            with patch.dict(sys.modules, {"datasets": module}):
                loaded = load_dataset_source(source)

        self.assertIs(loaded, load_dataset.return_value)
        load_dataset.assert_called_once_with(
            "json",
            data_files={"validation": str(path.resolve())},
            split="validation",
            cache_dir="cache",
        )

    def test_local_dataset_selects_requested_split(self) -> None:
        """从 load_from_disk 返回的 DatasetDict 中选择请求的 split。"""

        selected = object()
        dataset_dict = _FakeDatasetDict(test=selected)
        module, _, load_from_disk = _datasets_module(loaded_dataset=dataset_dict)
        with TemporaryDirectory() as directory:
            source = LocalDatasetSource(directory, split="test")
            with patch.dict(sys.modules, {"datasets": module}):
                loaded = load_dataset_source(source)

        self.assertIs(loaded, selected)
        load_from_disk.assert_called_once_with(str(Path(directory).resolve()))

    def test_local_dataset_dict_requires_existing_split(self) -> None:
        """拒绝没有指定 split 或请求未知 split 的本地 DatasetDict。"""

        dataset_dict = _FakeDatasetDict(train=object())
        module, _, _ = _datasets_module(loaded_dataset=dataset_dict)
        with TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"datasets": module}):
                with self.assertRaisesRegex(ValueError, "split is required"):
                    load_dataset_source(LocalDatasetSource(directory))
                with self.assertRaisesRegex(KeyError, "validation"):
                    load_dataset_source(
                        LocalDatasetSource(directory, split="validation")
                    )

    def test_import_qcomp_does_not_load_datasets(self) -> None:
        """在独立解释器中确认核心包导入不会加载 datasets。"""

        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import qcomp; assert 'datasets' not in sys.modules",
            ],
            cwd=Path(__file__).parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
