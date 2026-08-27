# Checkpoint formats

## Single module

每个模块目录包含 `cores.safetensors` 和 `manifest.json`。manifest 的格式标识保持
`qwen3-tn-tt-matrix-v1`，核心字段是 `module_path`、`spec`、`core_dtype`、
`checkpoint_bytes` 与 `cores_sha256`。旧 v1 manifest 没有 SHA 字段时仍可读取；
新写入的文件始终记录并校验 SHA-256。

## Module set

一个可安装集合必须有 `index.json`，格式为 `qwen3-tn-tt-module-set-v1`。每个条目
显式记录 `module_path`、相对 checkpoint 目录和 core SHA-256。相对路径不得越出
module-set 根目录，不允许重复 module path。

## Training checkpoint

`checkpoint-N/` 包含：

- `tt_modules/index.json` 与 FP32 canonical cores；
- `training_state.pt`：optimizer、scheduler 和 Python/Torch/CUDA RNG；
- `trainer_state.json`：step、epoch、batch、完整训练配置、backend 和签名 metadata。

optimizer state 不允许跨 backend 恢复。`final/` 是独立 module set，使用 BF16 cores，
可跨 backend 安装。
