# backends

`backends` 把不同的张量网络计算库统一包装成两个标准接口：

- __分解__：接收稠密权重和 representations 层定义的张量网络结构配置，调用具体计算库的分解算法，输出规范张量网络 artifact。
- __构造可执行层__：接收张量网络 artifact，构造并返回 `nn` 层约定的可执行 PyTorch 模块。

上层代码只跟 `TensorNetworkBackend` 打交道，不必关心底层用的是哪个库。


## 目录结构

```text
backends/
├── base.py                 # 通用 backend 接口、能力和环境探测结果
├── registry.py             # 按 Provider 与表示查找并延迟加载适配器
├── native/
│   └── mpo.py              # 原生 PyTorch + MPO
├── tensorly/
│   └── mpo.py              # TensorLy + MPO
├── torchtt/
│   └── mpo.py              # torchTT + MPO
└── cutensornet/
    └── mpo.py              # cuTensorNet + MPO
```

目录的第一维是计算库 Provider，文件名是张量网络表示。以后只有实现了真实组合时才新增文件；未实现的组合不注册，也不提供 fallback。

日常调用只需要两个查询函数：

- `get_backend`：根据 Provider 和表示延迟导入 qcomp 适配模块并创建适配器；第三方计算库在实际分解或 Linear 构造时导入。
- `list_backends`：批量评测时，列出指定表示已经注册的 Provider。

`TensorNetworkBackend`、`BackendCapabilities` 和 `BackendProbe` 是实现或批量检查后端时使用的类型，基础调用不需要直接创建它们。

## Backend 与 Linear 的关系

Backend 是可以复用的工厂，不保存某个模型层的 cores。Linear 是放入模型的 `torch.nn.Module`，每个实例独立持有自己的参数和运行资源。

```text
一个 CuTensorNetMPOBackend
├── 构造 CuTensorNetMPOLinear（q_proj 的 cores 和收缩计划）
├── 构造 CuTensorNetMPOLinear（k_proj 的 cores 和收缩计划）
└── 构造 CuTensorNetMPOLinear（v_proj 的 cores 和收缩计划）
```

Backend 构造完成后不拥有这些 Linear；模型负责保存 Linear，Linear 自己负责导出最新 artifact 和释放运行资源。

## 当前支持能力

| Provider | 表示 | 分解 | 推理 | 训练 |
|---|---|---:|---:|---:|
| `native` | MPO | 是 | 是 | 是 |
| `tensorly` | MPO | 是 | 是 | 是 |
| `torchtt` | MPO | 是 | 是 | 是 |
| `cutensornet` | MPO | 否 | 是 | 否 |

## 基础调用示例

下面的例子使用 native 后端完成一次稠密权重分解、模型层构造、推理和 artifact 导出：

```python
import torch

from qcomp import MPOSpec, get_backend

weight = torch.randn(8, 8)
inputs = torch.randn(2, 8)
spec = MPOSpec(
    in_modes=(2, 4),
    out_modes=(2, 4),
    ranks=(1, 4, 1),
)

backend = get_backend(provider="native", representation="mpo")
mpo_artifact = backend.decompose(weight, spec)
linear = backend.build_linear(mpo_artifact, trainable=True)
outputs = linear(inputs)

updated_mpo_artifact = linear.export_artifact()
linear.close()
```

`mpo_artifact` 是不同后端之间交换 MPO 数据的规范形式。把 `provider` 改为 `tensorly` 或 `torchtt` 时，整体调用流程保持不变。`cutensornet` 不提供分解，需要接收其他后端生成或从磁盘加载的 artifact，并在 CUDA 设备上构造仅推理 Linear。

直接调用后端不支持的操作不会切换到其他后端。基类会抛出 `NotImplementedError`，例如 `CuTensorNetMPOBackend.decompose()` 的错误信息为 `cutensornet/mpo does not support decomposition`。

## 批量评测多个后端

批量评测时可以使用 `list_backends()`、`capabilities` 和 `probe()`。下面的代码接续基础示例中的 `weight` 和 `spec`：

```python
from qcomp import get_backend, list_backends

for provider in list_backends("mpo"):
    backend = get_backend(provider, "mpo")
    status = backend.probe()
    if not status.available:
        continue
    if backend.capabilities.decomposition:
        mpo_artifact = backend.decompose(weight, spec)
```

`capabilities` 描述后端自身支持哪些操作，与当前机器无关。`probe()` 检查依赖和硬件在当前环境中是否可用，并在不可用时给出原因。因此，一个后端可以支持推理，但因为当前机器缺少依赖而暂时不可用。

## 新增计算后端

新增计算库 Provider 时，只实现它真实支持的张量网络表示。假设要增加 `newlib` 的 MPO 支持，目录为：

```text
backends/
└── newlib/
    ├── __init__.py
    └── mpo.py
```

`newlib/mpo.py` 需要实现两个对象：

- `NewLibMPOLinear`：表示模型中的一个 MPO Linear，持有该层的 cores 或第三方库对象。它继承 `MPOLinearBase`；直接保存项目规范 cores 时可继承 `CanonicalMPOLinearBase`，并实现实际 forward 收缩。
- `NewLibMPOBackend`：继承 `TensorNetworkBackend[MPOSpec]`，声明 `provider = "newlib"`、`representation = "mpo"` 和实际 capabilities，并实现支持的 `decompose()`、`build_linear()`、`probe()` 与版本查询。

后端分解结果必须转换成规范 MPO artifact：

```python
from ...representations import make_mpo_artifact

mpo_artifact = make_mpo_artifact(spec, cores)
```

模型层构造时使用 `parse_mpo_artifact()` 读取相同格式。可选依赖在模块函数或对象构造阶段导入，不在 `qcomp` 顶层导入。

实现完成后，在 `registry.py` 注册真实组合：

```python
_BACKENDS = {
    # 现有注册项
    ("newlib", "mpo"): (".newlib.mpo", "NewLibMPOBackend"),
}
```

对应测试至少覆盖 registry 查询、依赖延迟加载、capabilities、数值一致性，以及该后端实际支持的分解、训练和推理路径。

## 新增张量网络表示类型

新增表示的结构配置、artifact、重建注册、模型层和后端接入步骤见[扩展指南](../../../docs/extending.md#新增张量网络表示)。

浮点类型由 artifact tensors 保留；分解时需要转换类型可设置 `decomposition_dtype`，见 [workflows](../workflows/README.md)。
