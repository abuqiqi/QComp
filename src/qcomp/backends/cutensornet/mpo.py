"""使用 NVIDIA cuTensorNet 实现仅推理的 MPO 收缩。

本模块把规范 MPO artifact 转换为按输入形状、设备和数据类型缓存的 cuTensorNet
Network 计划，执行真实 CUDA 收缩并显式释放规划资源。cuQuantum 只在需要时加载。

主要内容：
- ``_package_available``、``_network_types``：探测并延迟导入 cuQuantum。
- ``CuTensorNetMPOLinear``：有状态模型层，持有 cores、收缩计划并执行 forward。
- ``CuTensorNetMPOBackend``：无状态适配器，报告仅推理能力并负责模型层构造。
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from typing import Any, Callable

import torch
from torch import Tensor

from ...nn.mpo import CanonicalMPOLinearBase, MPOLinearBase
from ...representations import MPOSpec, TensorNetworkArtifact
from ..base import (
    BackendCapabilities,
    BackendProbe,
    TensorNetworkBackend,
)


def _package_available() -> bool:
    """返回 cuTensorNet Python 包是否可导入。"""

    return importlib.util.find_spec("cuquantum") is not None


def _network_types() -> tuple[type[Any], type[Any], Any]:
    """仅在选择当前后端时导入 cuTensorNet network 类型。"""

    from cuquantum import ComputeType
    from cuquantum.tensornet import Network, NetworkOptions

    return Network, NetworkOptions, ComputeType


class CuTensorNetMPOLinear(CanonicalMPOLinearBase):
    """使用已规划的 cuTensorNet networks 收缩固定的规范 MPO cores。"""

    def __init__(self, artifact: TensorNetworkArtifact) -> None:
        """创建仅用于推理的 cuTensorNet 运行时。

        参数：
            artifact: 规范 MPO artifact。
        """

        super().__init__(artifact, trainable=False)
        self._networks: dict[
            tuple[int, str, torch.dtype],
            tuple[Any, Tensor],
        ] = {}

    @property
    def backend_name(self) -> str:
        """返回稳定的 cuTensorNet 后端名称。"""

        return "cutensornet"

    def _free_networks(self) -> None:
        """释放全部已规划的 cuTensorNet network 对象。"""

        for network, _ in self._networks.values():
            network.free()
        self._networks.clear()

    def close(self) -> None:
        """释放缓存的 cuTensorNet 资源。"""

        self._free_networks()

    def _apply(
        self,
        function: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "CuTensorNetMPOLinear":
        """在移动模块张量前清除设备相关的收缩计划。

        参数：
            function: ``torch.nn.Module`` 应用的张量转换函数。
            recurse: 是否同时转换子模块。

        返回：
            转换后的当前模块。
        """

        self._free_networks()
        return super()._apply(function, recurse)

    def _network_operands(self, inputs: Tensor) -> list[Any]:
        """构造 cuTensorNet 交错 operands 和整数 mode 标签。

        参数：
            inputs: 张量化后的输入缓冲区。

        返回：
            ``cuquantum.tensornet.Network`` 接受的 operands。
        """

        order = self.spec.order
        input_modes = list(range(1, order + 1))
        output_modes = list(range(order + 1, 2 * order + 1))
        rank_modes = list(range(2 * order + 1, 3 * order + 2))
        operands: list[Any] = [inputs, [0, *input_modes]]
        for index, core in enumerate(self.cores):
            operands.extend(
                [
                    core,
                    [
                        rank_modes[index],
                        output_modes[index],
                        input_modes[index],
                        rank_modes[index + 1],
                    ],
                ]
            )
        operands.append([0, *output_modes])
        return operands

    def _get_network(self, flat_input: Tensor) -> tuple[Any, Tensor]:
        """获取与指定 token 形状完全匹配的收缩计划。

        参数：
            flat_input: 形状为 [tokens, in_features] 的输入。

        返回：
            已规划的 network 及其可变输入缓冲区。
        """

        key = (flat_input.shape[0], str(flat_input.device), flat_input.dtype)
        cached = self._networks.get(key)
        if cached is not None:
            return cached
        Network, NetworkOptions, ComputeType = _network_types()
        buffer = torch.empty(
            (flat_input.shape[0], *self.spec.in_modes),
            device=flat_input.device,
            dtype=flat_input.dtype,
        )
        network = Network(
            *self._network_operands(buffer),
            options=NetworkOptions(
                compute_type=ComputeType.COMPUTE_32F,
                blocking="auto",
            ),
        )
        network.contract_path()
        self._networks[key] = (network, buffer)
        return network, buffer

    def _contract(self, flat_input: Tensor) -> Tensor:
        """在 CUDA 上使用 cuTensorNet 收缩输入。

        参数：
            flat_input: 位于 CUDA、形状为 [tokens, in_features] 的输入。

        返回：
            形状为 [tokens, out_features] 的输出。
        """

        if flat_input.device.type != "cuda":
            raise ValueError("cutensornet requires CUDA inputs")
        network, buffer = self._get_network(flat_input)
        buffer.copy_(flat_input.reshape_as(buffer))
        return network.contract().reshape(
            flat_input.shape[0],
            self.spec.out_features,
        )


class CuTensorNetMPOBackend(TensorNetworkBackend[MPOSpec]):
    """提供仅用于推理的 cuTensorNet 收缩能力。"""

    provider = "cutensornet"
    representation = "mpo"

    @property
    def version(self) -> str:
        """返回已安装的 cuQuantum Python 版本。"""

        return version("cuquantum-python-cu12")

    @property
    def capabilities(self) -> BackendCapabilities:
        """返回 cuTensorNet 后端能力。"""

        return BackendCapabilities(
            decomposition=False,
            inference=True,
            training=False,
            requires_cuda=True,
        )

    def probe(self) -> BackendProbe:
        """检查 cuTensorNet 和 CUDA 是否可用。"""

        package_available = _package_available()
        available = package_available and torch.cuda.is_available()
        reason = None
        if not package_available:
            reason = "missing package: cuquantum-python-cu12"
        elif not torch.cuda.is_available():
            reason = "CUDA is not available"
        return BackendProbe(available=available, reason=reason)

    def build_linear(
        self,
        artifact: TensorNetworkArtifact,
        *,
        trainable: bool,
    ) -> MPOLinearBase:
        """构造仅用于推理的 cuTensorNet MPO 线性层模块。

        参数：
            artifact: 位于 CUDA 设备的规范 MPO artifact。
            trainable: 必须为 false，因为 cuTensorNet 仅支持推理。

        返回：
            cuTensorNet MPO 运行时。
        """

        if trainable:
            raise ValueError("cutensornet does not support training")
        return CuTensorNetMPOLinear(artifact)
