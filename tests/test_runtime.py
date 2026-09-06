"""验证所有外部 I/O 共用的项目 runtime 配置。

本模块检查默认 TOML 的项目相对路径、严格离线环境、单次调用覆盖、配置只读语义和
命令注册。测试不会访问网络，也不会加载真实模型或数据集。

主要内容：
- ``RuntimeConfigTests``：验证默认配置、覆盖、校验和懒应用行为。
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from qcomp.runtime import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    PROJECT_ROOT,
    configure_runtime,
    load_runtime_config,
)


class RuntimeConfigTests(unittest.TestCase):
    """验证模型、数据集和 lm-eval 使用同一份 runtime 配置。"""

    def test_default_runtime_is_project_relative_and_offline(self) -> None:
        """默认配置指向工作区本地资源并设置所有离线环境变量。"""

        with patch.dict(os.environ, {}, clear=True):
            runtime = configure_runtime()

            expected_home = (PROJECT_ROOT / "../../datasets/huggingface").resolve()
            self.assertEqual(
                runtime.model_name_or_path,
                str((PROJECT_ROOT / "../../models/Qwen3-8B").resolve()),
            )
            self.assertEqual(runtime.huggingface_home, expected_home)
            self.assertEqual(runtime.datasets_cache, expected_home / "datasets")
            self.assertTrue(runtime.offline)
            self.assertEqual(os.environ["HF_HOME"], str(expected_home))
            self.assertEqual(
                os.environ["HF_DATASETS_CACHE"],
                str(expected_home / "datasets"),
            )
            for name in (
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            ):
                self.assertEqual(os.environ[name], "1")

    def test_loading_runtime_does_not_modify_environment(self) -> None:
        """只解析配置时不产生目录或进程环境副作用。"""

        with patch.dict(os.environ, {}, clear=True):
            runtime = load_runtime_config()
            self.assertTrue(runtime.offline)
            self.assertNotIn("HF_HOME", os.environ)

    def test_call_overrides_do_not_rewrite_toml(self) -> None:
        """一次调用覆盖模型和缓存目录，但保持 runtime.toml 不变。"""

        original = DEFAULT_RUNTIME_CONFIG_PATH.read_bytes()
        with TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            relative_home = os.path.relpath(directory, PROJECT_ROOT)
            with patch.dict(os.environ, {}, clear=True):
                runtime = configure_runtime(
                    model_name_or_path="Qwen/temporary-model",
                    huggingface_home=relative_home,
                )
            self.assertEqual(runtime.model_name_or_path, "Qwen/temporary-model")
            self.assertEqual(runtime.huggingface_home, Path(directory).resolve())
            self.assertEqual(runtime.datasets_cache, Path(directory) / "datasets")

        self.assertEqual(DEFAULT_RUNTIME_CONFIG_PATH.read_bytes(), original)

    def test_runtime_rejects_absolute_cache_paths(self) -> None:
        """拒绝破坏项目相对路径约定的绝对缓存配置。"""

        with self.assertRaisesRegex(ValueError, "relative"):
            configure_runtime(huggingface_home="/tmp/huggingface")

    def test_import_qcomp_does_not_apply_runtime(self) -> None:
        """独立解释器导入核心包时不读取配置或修改离线环境。"""

        environment = dict(os.environ)
        environment.pop("HF_HOME", None)
        environment["PYTHONPATH"] = "src"
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; import qcomp; assert 'HF_HOME' not in os.environ",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_pyproject_does_not_register_experiment_script(self) -> None:
        """确认具体 sensitivity 实验脚本不作为包内公共命令安装。"""

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        self.assertNotIn("scripts", pyproject["project"])


if __name__ == "__main__":
    unittest.main()
