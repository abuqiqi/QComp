# `representations` 代码结构

`qcomp.representations` 定义张量网络结构的数据格式和与计算库无关的数学操作。
不同 backend 将分解结果转换成这里规定的统一格式；`nn`、`evaluation` 和 `storage`
只依赖该格式，不需要了解结果来自 TensorLy、torchTT 或其他计算库。

当前只实现 MPO。实际分解算法位于 `backends`，本目录负责描述 MPO 结构、校验结果、
转换 artifact 和重建稠密权重。

## 目录结构

```text
representations/
├── README.md     # 当前目录的职责、数据格式和扩展方式
├── __init__.py   # 导出公共接口
├── artifact.py   # 通用 TensorNetworkArtifact 数据容器
├── registry.py   # 根据表示名称选择稠密重建函数
└── mpo.py        # MPO 结构、canonical core 格式和数学操作
```

## 主要公共对象

- `TensorNetworkArtifact`：保存表示名称、格式版本、结构元数据和命名张量的通用数据容器，定义在 [`artifact.py`](artifact.py)。
- `MPOSpec`：描述 MPO 的输出 modes、输入 modes 和 ranks，定义在 [`mpo.py`](mpo.py)。
- `make_mpo_artifact()`：将符合项目格式的 MPO cores 封装成 `TensorNetworkArtifact`。
- `parse_mpo_artifact()`：从通用 artifact 中读取并校验 `MPOSpec` 和 MPO cores。
- `reconstruct_tensor()`：根据 `artifact.representation` 选择对应算法并重建稠密张量，定义在 [`registry.py`](registry.py)。

## `MPOSpec` 和 `TensorNetworkArtifact` 的区别

`MPOSpec` 是分解前的结构配置，只描述希望得到什么形状的 MPO，不保存实际参数。
`TensorNetworkArtifact` 是分解后的结果，包含实际 cores 及其结构元数据。

```text
MPOSpec
    out_modes、in_modes、ranks

TensorNetworkArtifact
    representation、format_version、metadata、tensors
```

MPO artifact 当前采用以下格式：

```text
representation = "mpo"
metadata       = out_modes、in_modes、ranks
tensors        = core_0、core_1、...
core layout    = [left_rank, out_mode, in_mode, right_rank]
```

这里的 canonical 表示 qcomp 规定的统一 core 布局，不表示数学上的左正交或右正交形式。

## 基础用法

backend 分解得到通用 artifact 后，可以直接使用 representations 层检查其结构或重建
稠密权重：

```python
import torch

from qcomp import MPOSpec, get_backend, parse_mpo_artifact, reconstruct_tensor

weight = torch.randn(8, 8)
spec = MPOSpec(out_modes=(2, 4), in_modes=(2, 4), ranks=(1, 4, 1))
backend = get_backend(provider="native", representation="mpo")
tn_artifact = backend.decompose(weight, spec)

parsed_spec, cores = parse_mpo_artifact(tn_artifact)
reconstructed = reconstruct_tensor(tn_artifact)

print(parsed_spec.ranks)       # (1, 4, 1)
print(len(cores))              # 2
print(reconstructed.shape)     # torch.Size([8, 8])
```

## 新增张量网络表示

以新增 Tucker 为例：

1. 新建 `representations/tucker.py`，定义 Tucker 的结构配置、校验、artifact 构造、解析和稠密重建函数。
2. 在 `registry.py` 中注册 `"tucker"` 对应的重建函数。
3. 在 `representations/__init__.py` 中导出需要公开的接口。
4. 在需要支持 Tucker 的 backend 中实现分解和模型层构造。

`TensorNetworkArtifact` 保持通用，不需要为每种张量网络结构新建一套数据容器。
