"""组织 torchTT Provider 的张量网络适配器。

本包按照张量网络表示拆分 torchTT 实现，使每个模块对应一个明确的 Provider 与表示
组合，并保持可选依赖延迟加载。

主要内容：
- ``mpo``：提供 torchTT 与 MPO 组合的分解、训练和推理实现。
"""
