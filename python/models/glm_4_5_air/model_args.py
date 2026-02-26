"""GLM-4.5-Air model configuration."""

from dataclasses import dataclass
from typing import Literal
from tilert.models.deepseek_v3_2.model_args import ModelArgs

__all__ = ["ModelArgsGLM4P5Air"]


@dataclass
class ModelArgsGLM4P5Air(ModelArgs):
    """
    GLM-4.5-Air configuration.
    Hybrid: Layer 0 is Dense (BF16), layers 1-45 are MoE.
    Uses GQA (Grouped Query Attention): 96 query heads, 8 KV heads.
    """

    arch_name = "glm_4_5_air"

    max_batch_size: int = 1
    max_seq_len: int = 32768
    dtype: Literal["bf16", "fp8"] = "bf16"  # BF16, not FP8

    # Architecture
    vocab_size: int = 151552
    dim: int = 4096
    inter_dim: int = 10944
    moe_inter_dim: int = 2048
    n_layers: int = 46
    n_dense_layers: int = 1
    
    # GQA Configuration
    n_heads: int = 96  # Total query heads
    n_kv_heads: int = 8  # KV heads (GQA)
    head_dim: int = 128  # Dimension per head
    
    # Q/K/V dimensions for GQA
    qk_nope_head_dim: int = 128  # Query/key dim without RoPE
    qk_rope_head_dim: int = 64   # RoPE dimension
    v_head_dim: int = 128        # Value dimension

    # MoE
    n_routed_experts: int = 128
    n_activated_experts: int = 8

    # GQA uses standard attention (no LoRA compression)
    q_lora_rank: int = 0
    kv_lora_rank: int = 0

    # No quantization scales
    scale_fmt: None = None

    eps: float = 1e-5  # From config.json
    block_size: int = 128
    
    # RoPE settings from config.json
    rope_theta: float = 1000000.0
    rope_factor: float | None = None
    original_seq_len: int = 131072
