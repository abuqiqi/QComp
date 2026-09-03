"""组织 native Provider 的张量网络适配器。

本包按照张量网络表示拆分原生 PyTorch 实现，使 Provider 目录只包含 native 实际支持
的表示，并避免把不同表示的算法混在同一文件中。

主要内容：
- ``mpo``：提供 native 与 MPO 组合的分解、训练和推理实现。
"""
