# `qcomp` 源码结构

本目录是 `qcomp` Python 包的源码入口。`model.py` 和 `storage.py` 提供跨模块使用的
通用功能；各子目录分别负责张量网络表示、PyTorch 模型层、计算后端和评测。

当前代码支持单个无 bias Linear 的 MPO 分解、模型层替换、训练、推理、评测以及
artifact 保存和加载。

## 目录结构

```text
qcomp/
├── README.md          # 源码结构和职责说明
├── __init__.py        # 导出常用公共接口
├── model.py           # 列出、查找、替换和恢复模型中的 Linear
├── storage.py         # 保存和加载 TensorNetworkArtifact
├── representations/   # 张量网络数据格式和数学操作
├── nn/                # 张量网络 PyTorch 模型层的基类和公共行为
├── backends/          # 不同计算库的分解和具体模型层实现
└── evaluation/        # 压缩指标及分解、训练、推理计时
```

## 顶层文件

### `model.py`

`model.py` 负责操作 PyTorch 模型结构。

主要公共对象：

- `list_linears()`：列出模型中具有模块路径的无 bias `torch.nn.Linear`。
- `find_linear()`：按模块路径取得一个无 bias Linear。
- `replace_linear()`：将目标 Linear 替换为 backend 已经构造好的 `TensorNetworkLinear`。
- `restore_linear()`：关闭压缩模型层并恢复原始 Linear。
- `LinearReplacement`：保存一次替换所需的目标路径、原始层和压缩层。

### `storage.py`

`storage.py` 负责持久化 representations 层定义的通用
`TensorNetworkArtifact`，即张量网络分解结果的数据容器。

主要公共函数：

- `save_artifact()`：保存表示名称、格式版本、元数据和命名张量。
- `load_artifact()`：加载上述数据，并恢复为 `TensorNetworkArtifact`。

### `__init__.py`

`__init__.py` 汇总常用公共接口，使调用者可以直接从 `qcomp` 导入。可选 backend 依赖仍保持懒加载，不会因为导入 `qcomp` 而全部加载。

## 相关目录

- [`representations/`](representations/README.md)：定义 `TensorNetworkArtifact`、`MPOSpec`、canonical MPO 格式和稠密重建。
- [`nn/`](nn/README.md)：定义 `TensorNetworkLinear`、MPO Linear 基类和公共模型层行为。
- [`backends/`](backends/README.md)：实现 native、TensorLy、torchTT 和 cuTensorNet 的 MPO 适配器与具体模型层。
- [`evaluation/`](evaluation/README.md)：计算压缩指标，并测量分解、推理和训练 step 时间。
