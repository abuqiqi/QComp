# nn

`nn` 提供张量网络模型层的基类和公共行为。其中最上层的 `TensorNetworkLinear` 继承 PyTorch 的 `torch.nn.Module`，因此具体模型层可以直接放入 PyTorch 模型并参与训练或推理。

`nn` 不实现某个计算库的完整模型层。`backends` 中的 `NativeMPOLinear`、`TensorLyMPOLinear` 等具体模型层继承这里的基类，并实现各自的张量收缩。模型替换、训练和 evaluation 可以通过 `TensorNetworkLinear` 使用这些具体实现，而不依赖某个 Provider。

当前只实现无 bias 的 MPO Linear。权重分解和计算库选择由 `backends` 负责，张量网络的数据格式和数学定义由 `representations` 负责，本目录不重复这些功能。

## 目录结构

```text
nn/
├── __init__.py  # 导出公共模型层类型
├── base.py      # 所有张量网络表示共用的 PyTorch 模块接口
└── mpo.py       # MPO Linear 的公共行为和 canonical core 容器
```

`base.py` 和 `mpo.py` 两个文件处于不同的抽象层级：

| 文件 | 适用范围 | 负责的内容 |
|---|---|---|
| `base.py` | 各种张量网络结构表示 | 定义 `TensorNetworkLinear`，统一 artifact 导出和运行时资源释放接口 |
| `mpo.py` | 仅 MPO | 处理输入形状、MPO artifact 导出、canonical cores 参数注册 |

如果以后加入其他张量网络结构，可以新增 `nn/[structure].py` 并继承 `TensorNetworkLinear`，不需要修改通用的 `base.py`。这种拆分避免不同后端分别实现相同的输入整形和 artifact 导出代码。

## 主要公共对象

- `TensorNetworkLinear`：所有张量网络 Linear 的通用基类。它继承 `torch.nn.Module`，并要求模型层能够导出张量网络分解结果的统一数据容器 `TensorNetworkArtifact`。该容器定义在 [`representations/artifact.py`](../representations/artifact.py)；`close()` 为需要显式释放运行时资源的实现提供统一入口。
- `MPOLinearBase`：所有 MPO Linear 的公共基类。它保存 `MPOSpec`，统一检查输入最后一维，并负责把任意批次形状展平后交给具体后端收缩。
- `CanonicalMPOLinearBase`：使用项目 canonical MPO cores 作为 `nn.Parameter` 的中间基类。native 和 cuTensorNet 模型层在此基础上实现各自的收缩逻辑。

这些类型主要供 backend 实现复用，但不只服务于 backend。backend 负责创建具体模型层；模型替换、训练、推理和 evaluation 则通过 `TensorNetworkLinear` 使用创建后的模型层。

当前继承关系如下：

```text
torch.nn.Module                         (PyTorch)
└── TensorNetworkLinear                (src/qcomp/nn/base.py)
    └── MPOLinearBase                  (src/qcomp/nn/mpo.py)
        ├── CanonicalMPOLinearBase     (src/qcomp/nn/mpo.py)
        │   ├── NativeMPOLinear        (src/qcomp/backends/native/mpo.py)
        │   └── CuTensorNetMPOLinear   (src/qcomp/backends/cutensornet/mpo.py)
        ├── TensorLyMPOLinear          (src/qcomp/backends/tensorly/mpo.py)
        └── TorchTTMPOLinear           (src/qcomp/backends/torchtt/mpo.py)
```

TensorLy、torchTT 等第三方模型层使用自己的容器构造并管理张量网络参数，所以二者直接继承 `MPOLinearBase`。native 和 cuTensorNet 没有这种参数容器，因此二者都由 qcomp 使用 canonical core 布局和 `nn.ParameterList` 保存参数，通过 `CanonicalMPOLinearBase` 复用 artifact 解析、参数注册和导出逻辑。

## 新增张量网络表示

新增表示时，只在存在共享的模型层行为时增加对应文件。例如加入 Tucker：

1. 在 `representations/tucker.py` 定义数据格式、校验和重建逻辑。
2. 在 `nn/tucker.py` 定义 `TuckerLinearBase`，集中实现 Tucker 后端共有的输入处理和 artifact 导出行为。
3. 在需要支持的 `backends/<provider>/tucker.py` 中实现 Provider 专用的分解和模型层。
4. 让具体模型层继承 `TuckerLinearBase`，并由对应 backend 的 `build_linear()` 创建。

没有多个 backend 共用的行为时，不需要提前创建额外的中间基类。
