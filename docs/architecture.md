# Architecture

代码依赖保持单向：

```text
tt → backends → checkpoint + model
   → data + training + evaluation
   → workflows + benchmarking → cli
```

`tt` 不知道模型、文件格式或 backend。`backends` 只把 canonical cores 变成线性层；
公共基类统一 core 生命周期、dtype、输入校验和 activation checkpointing，各适配器只
实现 contraction。registry 区分“已注册”和“当前环境可用”，backend capabilities 在
构建前拒绝不支持的训练或 checkpointing 组合。
`checkpoint` 只读写 cores 和显式 module-set index；`model` 负责按 `module_path`
验证、替换和回滚模块。训练器只接收已安装模型与 DataLoader，不负责加载数据或定位层。

数据流为：Dense weight → TT-SVD → canonical cores → module-set index → runtime backend
→ TT-only training/evaluation。Qwen 路径和具体数据集仅出现在 `configs/`。
