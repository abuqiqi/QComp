# TT-matrix conventions

对于形状为 `[M, N]` 的矩阵，选择 `M=∏m_k`、`N=∏n_k`。先将矩阵重排为交错
物理模态 `(m_1 n_1, …, m_d n_d)`，再做逐级 TT-SVD。

第 `k` 个 canonical core 的布局固定为：

```text
[r_k, m_k, n_k, r_{k+1}]
```

`r_0=r_d=1`。`TTMatrixSpec.ranks` 始终包含两个边界 rank，因此长度为
`order+1`。`reconstruct_matrix` 仅用于验证；运行时 backend 直接 contraction，
不会重建 Dense 权重。`slice_bond` 同时裁剪相邻两个 core，主要用于研究 sweep。

