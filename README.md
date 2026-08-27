# Qwen3 TT Compression

这是一个面向 causal language model 的 TT-matrix 研究仓库。核心代码不依赖 Qwen
层路径或特定数据集：模型目标由 JSON 中的 `module_path` 描述，checkpoint 使用
backend-neutral canonical cores，运行时可选择原生 PyTorch 或 TensorLy-Torch。

## 安装

```bash
cd /home/xls/workspace/projects/qwen3-tn-compression
pip install -e .
```

TensorLy-Torch 是可选后端：

```bash
pip install -e ".[tensorly]"
```

## 稳定 API

包根目录只导出五个稳定入口：

```python
from qwen3_tn import (
    TTMatrixSpec,
    TTTarget,
    available_tt_backends,
    create_tt_linear,
    tt_svd_matrix,
)
```

更高层能力使用模块导入：

```python
from qwen3_tn.model import install_tt_modules, TTModulePatch
from qwen3_tn.data import JsonlDocumentSource, prepare_causal_lm_data
from qwen3_tn.training import TTFineTuneConfig, finetune_causal_lm
from qwen3_tn.workflows import run_finetune_experiment, run_rank_sweep
```

module set 必须包含 `index.json`。安装层不会递归扫描目录，也不会猜测 manifest
归属：

```python
installed = install_tt_modules(
    model,
    "artifacts/decompositions/qwen3-8b-down-proj-r2048/index.json",
    tt_backend="native",
    trainable=True,
    core_dtype=torch.float32,
)
```

`TTModulePatch` 用于临时评测或 sweep，退出上下文后自动恢复原 Dense 层。

## CLI

五个 CLI 都是库函数的薄入口，使用严格 JSON 配置：

```bash
qwen3-tn-decompose configs/compression/qwen3_8b_down_proj_r2048.json
qwen3-tn-finetune configs/experiments/qwen3_8b_wikitext_finetune.json
qwen3-tn-evaluate /path/to/evaluation_run.json
qwen3-tn-sweep configs/sweeps/qwen3_8b_layer0_down_proj.json
qwen3-tn-benchmark layer configs/experiments/qwen3_8b_backend_benchmark.json
```

训练始终冻结 backbone，只优化 FP32 TT cores；CUDA 前向默认 BF16 autocast。
TT contraction activation checkpointing 默认开启，模型级 gradient checkpointing
按配置显式启用。训练 checkpoint 含 optimizer、scheduler、循环位置和 RNG；最终
`final/index.json` 仅导出 BF16 canonical cores。

## 目录职责

- `src/qwen3_tn/tt/`：纯数学与 TT-SVD。
- `src/qwen3_tn/backends/`：原生与 TensorLy-Torch 运行后端。
- `checkpoint.py` / `model.py`：canonical checkpoint 与模型安装。
- `data/` / `training/` / `evaluation.py`：数据、TT-only 训练和单模型评测。
- `workflows/` / `benchmarking.py`：可组合研究流程与性能测试。
- `cli/`：只解析 JSON 并调用库函数。

详细依赖边界、数学约定和格式规范位于 `docs/`。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

未安装 TensorLy-Torch 时，native 全套测试照常运行，TensorLy 数值一致性测试跳过。

