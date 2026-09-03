"""组织 TensorLy Provider 的张量网络适配器。

本包按照张量网络表示拆分 TensorLy 实现，可选依赖只在 registry 选择对应 Provider
与表示组合时加载。

主要内容：
- ``mpo``：提供 TensorLy 与 MPO 组合的分解、训练和推理实现。
"""
