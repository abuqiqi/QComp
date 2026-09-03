"""组织 cuTensorNet Provider 的张量网络适配器。

本包按照张量网络表示拆分 cuTensorNet 实现，并保持 cuQuantum 延迟导入，使核心包和
其他 Provider 不依赖 CUDA 运行环境。

主要内容：
- ``mpo``：提供 cuTensorNet 与 MPO 组合的真实 CUDA 推理实现。
"""
