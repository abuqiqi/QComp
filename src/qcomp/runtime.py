"""从项目 TOML 加载模型与 Hugging Face 本地运行配置。

本模块只使用标准库解析项目相对路径，并在模型、数据集或 lm-eval 首次执行 I/O 前
设置当前进程的 Hugging Face 缓存与离线环境。包导入本身不会读取配置或修改环境。

主要内容：
- ``RuntimeConfig``：保存解析后的模型来源、缓存目录和离线模式。
- ``load_runtime_config``：读取默认或显式指定的 TOML 文件。
- ``configure_runtime``：应用配置及可选的单次调用覆盖。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "config" / "runtime.toml"


@dataclass(frozen=True)
class RuntimeConfig:
    """保存已经解析为绝对路径的本地运行配置。"""

    model_name_or_path: str
    huggingface_home: Path
    datasets_cache: Path
    offline: bool


def _config_path(value: str | Path | None) -> Path:
    """解析默认或调用方指定的 runtime TOML 路径。

    参数：
        value: ``None``、绝对路径或相对于项目根目录的路径。

    返回：
        runtime TOML 的绝对路径。
    """

    if value is None:
        return DEFAULT_RUNTIME_CONFIG_PATH
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _project_path(value: object, *, name: str) -> Path:
    """把 TOML 中的项目相对路径解析为绝对路径。

    参数：
        value: TOML 中的路径值。
        name: 用于错误信息的配置名称。

    返回：
        相对于项目根目录的绝对路径。

    异常：
        ValueError: 路径不是非空字符串或使用绝对路径时抛出。
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{name} must be relative to the project root")
    return (PROJECT_ROOT / path).resolve()


def _model_source(value: object) -> str:
    """解析本地模型路径，同时保留 Hugging Face 模型 ID。

    参数：
        value: TOML 或调用覆盖提供的模型来源。

    返回：
        绝对本地路径或原始 Hugging Face 模型 ID。

    异常：
        ValueError: 模型来源不是非空字符串时抛出。
    """

    if not isinstance(value, str) or not value:
        raise ValueError("model.name_or_path must be a non-empty string")
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if value.startswith("."):
        return str((PROJECT_ROOT / path).resolve())
    return value


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    """读取项目运行配置但不修改进程环境。

    参数：
        path: 可选 runtime TOML；相对路径按项目根目录解析。

    返回：
        已规范路径并完成类型校验的运行配置。

    异常：
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 必需配置段、路径或离线字段无效时抛出。
    """

    config_path = _config_path(path)
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)
    model = values.get("model")
    huggingface = values.get("huggingface")
    if not isinstance(model, Mapping):
        raise ValueError("runtime config requires a [model] table")
    if not isinstance(huggingface, Mapping):
        raise ValueError("runtime config requires a [huggingface] table")
    offline = huggingface.get("offline")
    if not isinstance(offline, bool):
        raise ValueError("huggingface.offline must be a bool")
    return RuntimeConfig(
        model_name_or_path=_model_source(model.get("name_or_path")),
        huggingface_home=_project_path(
            huggingface.get("home"), name="huggingface.home"
        ),
        datasets_cache=_project_path(
            huggingface.get("datasets_cache"),
            name="huggingface.datasets_cache",
        ),
        offline=offline,
    )


def configure_runtime(
    path: str | Path | None = None,
    *,
    model_name_or_path: str | None = None,
    huggingface_home: str | Path | None = None,
    datasets_cache: str | Path | None = None,
) -> RuntimeConfig:
    """读取并应用当前 I/O 操作使用的本地运行配置。

    参数：
        path: 可选 runtime TOML 路径。
        model_name_or_path: 可选模型来源覆盖。
        huggingface_home: 可选项目相对 Hugging Face 根目录覆盖。
        datasets_cache: 可选项目相对 datasets 缓存覆盖。

    返回：
        合并调用覆盖后的运行配置。
    """

    configured = load_runtime_config(path)
    home = (
        _project_path(str(huggingface_home), name="huggingface.home")
        if huggingface_home is not None
        else configured.huggingface_home
    )
    cache = (
        _project_path(str(datasets_cache), name="huggingface.datasets_cache")
        if datasets_cache is not None
        else home / "datasets"
        if huggingface_home is not None
        else configured.datasets_cache
    )
    runtime = RuntimeConfig(
        model_name_or_path=(
            _model_source(model_name_or_path)
            if model_name_or_path is not None
            else configured.model_name_or_path
        ),
        huggingface_home=home,
        datasets_cache=cache,
        offline=configured.offline,
    )
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_DATASETS_CACHE"] = str(cache)
    offline_value = "1" if runtime.offline else "0"
    os.environ["HF_HUB_OFFLINE"] = offline_value
    os.environ["HF_DATASETS_OFFLINE"] = offline_value
    os.environ["TRANSFORMERS_OFFLINE"] = offline_value
    return runtime
