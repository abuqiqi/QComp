"""共享 Qwen3 的 MPO 拆分规则，供实验脚本和离线页面生成器使用。

本模块只依赖标准库，将矩阵维度转换为可序列化的结构，不加载模型或后端。
主要内容：
- ``qwen3_modes``：定义已支持特征维度的三核拆分。
- ``qwen3_mpo_spec_dict``：生成并校验统一内部 rank 的 MPO 结构。
- ``qwen3_projection_shapes``：从模型配置推导七类投影矩阵形状。
"""

from collections.abc import Mapping
from math import prod
from typing import Any


def qwen3_modes(size: int) -> tuple[int, int, int]:
    """返回 size 对应的三核 modes；不支持的维度明确报错。"""
    modes = {
        1024: (8, 8, 16),
        4096: (16, 16, 16),
        12288: (16, 16, 48),
        151936: (8, 16, 1187),
    }
    if size not in modes:
        raise ValueError(f"no Qwen3 MPO modes configured for feature size {size}")
    return modes[size]


def qwen3_mpo_spec_dict(out_features: int, in_features: int, rank: int) -> dict:
    """将矩阵维度和 rank 转换为 JSON spec，并检查两条内部 bond 上限。"""
    out_modes, in_modes = qwen3_modes(out_features), qwen3_modes(in_features)
    physical = [o * i for o, i in zip(out_modes, in_modes, strict=True)]
    if (
        type(rank) is not int
        or rank <= 0
        or any(rank > min(prod(physical[:j]), prod(physical[j:])) for j in (1, 2))
    ):
        raise ValueError(f"invalid Qwen3 MPO rank: {rank}")
    return dict(
        out_modes=list(out_modes), in_modes=list(in_modes), ranks=[1, rank, rank, 1]
    )


def qwen3_projection_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """从 Qwen3 config 推导 [out, in] 形状；拒绝不完整配置及带 bias 的投影。"""
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
    )
    if config.get("model_type") != "qwen3" or any(
        type(config.get(f)) is not int or config[f] <= 0 for f in fields
    ):
        raise ValueError("需要完整的 Qwen3 模型配置")
    if (
        config.get("attention_bias", False) is not False
        or config.get("mlp_bias", False) is not False
    ):
        raise ValueError("当前仅支持无 bias 的 Qwen3 投影")
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    if heads % kv_heads:
        raise ValueError("attention heads 必须是 KV heads 的整数倍")
    head = config.get("head_dim")
    if head is None:
        if hidden % heads:
            raise ValueError("无法推导 head_dim，请提供完整模型配置")
        head = hidden // heads
    if type(head) is not int or head <= 0:
        raise ValueError("head_dim 必须为正整数")
    return {
        "q_proj": (heads * head, hidden),
        "k_proj": (kv_heads * head, hidden),
        "v_proj": (kv_heads * head, hidden),
        "o_proj": (hidden, heads * head),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
