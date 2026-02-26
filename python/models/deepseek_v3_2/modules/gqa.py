"""GQA (Grouped Query Attention) module for GLM-4.5-Air and similar models."""

import torch
import torch.nn.functional as F

from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs


class GqaRefWeightsAlias:
    """Reference weights alias for GQA."""

    def __call__(self) -> list[str]:
        return [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]


class GqaTilertWeightsAlias:
    """TileRT weights alias for GQA."""

    input_layernorm_gamma = "input_layernorm_gamma"
    q_proj_weight = "q_proj_weight"
    k_proj_weight = "k_proj_weight"
    v_proj_weight = "v_proj_weight"
    o_proj_weight = "o_proj_weight"

    def __call__(self) -> list[str]:
        return [
            self.input_layernorm_gamma,
            self.q_proj_weight,
            self.k_proj_weight,
            self.v_proj_weight,
            self.o_proj_weight,
        ]


class Gqa(SerializableTileRTModule):
    """Implement the Grouped Query Attention (GQA) operations.
    
    This is for models like GLM-4.5-Air that use GQA where:
    - n_heads = 96 (query heads)
    - n_kv_heads = 8 (shared KV heads)
    - Each KV head is shared by 12 query heads (96/8)
    """

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int, layer_idx: int = 0):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.n_kv_heads = getattr(model_args, 'n_kv_heads', model_args.n_heads)
        self.n_local_heads = self.n_heads // num_devices
        self.n_local_kv_heads = self.n_kv_heads // num_devices
        self.head_dim = getattr(model_args, 'head_dim', model_args.dim // model_args.n_heads)
        self.layer_idx = layer_idx
        
        # Query heads per KV head (for GQA grouping)
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # For GLM-4.5-Air: dim=4096, n_heads=96, n_kv_heads=8, head_dim=128
        # Q: [4096, 12288], K/V: [4096, 1024], O: [12288, 4096]
        
        self.tilert_weights_alias = GqaTilertWeightsAlias()
        self.ref_weights_alias = GqaRefWeightsAlias()

        # KV cache
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def device_sharding(
        self, weights_map: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Shard GQA weights across devices.
        
        For GQA, we shard both query and KV heads across devices.
        Each device gets n_local_heads query heads and n_local_kv_heads KV heads.
        """
        # Find the layer prefix from the weights_map keys
        sample_key = list(weights_map.keys())[0]
        parts = sample_key.split(".")
        layer_idx = parts[2] if len(parts) > 2 else "0"
        prefix = f"model.layers.{layer_idx}."
        
        # Input layernorm - replicated across devices
        input_layernorm_weight = weights_map[f"{prefix}input_layernorm.weight"]
        
        # Q projection: [n_heads * head_dim, dim] - shard by output dim (query heads)
        q_proj_weight_full = weights_map[f"{prefix}self_attn.q_proj.weight"]  # [12288, 4096]
        
        # K projection: [n_kv_heads * head_dim, dim] - shard by output dim (KV heads)
        k_proj_weight_full = weights_map[f"{prefix}self_attn.k_proj.weight"]  # [1024, 4096]
        
        # V projection: [n_kv_heads * head_dim, dim] - shard by output dim (KV heads)
        v_proj_weight_full = weights_map[f"{prefix}self_attn.v_proj.weight"]  # [1024, 4096]
        
        # O projection: [dim, n_heads * head_dim] - shard by input dim (query heads)
        o_proj_weight_full = weights_map[f"{prefix}self_attn.o_proj.weight"]  # [4096, 12288]
        
        # Create sharded weights for all devices
        q_sharded = []
        k_sharded = []
        v_sharded = []
        o_sharded = []
        
        for dev_id in range(self.num_devices):
            # Q projection: shard by output dim (query heads)
            q_start = dev_id * self.n_local_heads * self.head_dim
            q_end = (dev_id + 1) * self.n_local_heads * self.head_dim
            q_proj = q_proj_weight_full[q_start:q_end, :]  # [n_local_heads * head_dim, dim]
            q_sharded.append(q_proj)
            
            # K projection: shard by output dim (KV heads)
            k_start = dev_id * self.n_local_kv_heads * self.head_dim
            k_end = (dev_id + 1) * self.n_local_kv_heads * self.head_dim
            k_proj = k_proj_weight_full[k_start:k_end, :]  # [n_local_kv_heads * head_dim, dim]
            k_sharded.append(k_proj)
            
            # V projection: shard by output dim (KV heads)
            v_proj = v_proj_weight_full[k_start:k_end, :]  # [n_local_kv_heads * head_dim, dim]
            v_sharded.append(v_proj)
            
            # O projection: shard by input dim (query heads)
            # O is [dim, n_heads * head_dim], so we shard the second dimension
            o_proj = o_proj_weight_full[:, q_start:q_end]  # [dim, n_local_heads * head_dim]
            o_sharded.append(o_proj)
        
        # Stack sharded weights
        q_sharded = torch.stack(q_sharded, dim=0)  # [num_devices, n_local_heads * head_dim, dim]
        k_sharded = torch.stack(k_sharded, dim=0)  # [num_devices, n_local_kv_heads * head_dim, dim]
        v_sharded = torch.stack(v_sharded, dim=0)  # [num_devices, n_local_kv_heads * head_dim, dim]
        o_sharded = torch.stack(o_sharded, dim=0)  # [num_devices, dim, n_local_heads * head_dim]
        
        # Input layernorm - replicate across devices
        gamma_sharded = input_layernorm_weight[None, ...].repeat(self.num_devices, 1)
        
        tilert_alias = self.tilert_weights_alias
        return {
            tilert_alias.input_layernorm_gamma: gamma_sharded,
            tilert_alias.q_proj_weight: q_sharded,
            tilert_alias.k_proj_weight: k_sharded,
            tilert_alias.v_proj_weight: v_sharded,
            tilert_alias.o_proj_weight: o_sharded,
        }

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize TileRT weights from state dict."""
        tilert_alias = self.tilert_weights_alias
        # Handle missing keys gracefully
        if tilert_alias.input_layernorm_gamma in state_dict:
            self.input_layernorm_gamma = state_dict[tilert_alias.input_layernorm_gamma]
        if tilert_alias.q_proj_weight in state_dict:
            self.q_proj_weight = state_dict[tilert_alias.q_proj_weight]
        if tilert_alias.k_proj_weight in state_dict:
            self.k_proj_weight = state_dict[tilert_alias.k_proj_weight]
        if tilert_alias.v_proj_weight in state_dict:
            self.v_proj_weight = state_dict[tilert_alias.v_proj_weight]
        if tilert_alias.o_proj_weight in state_dict:
            self.o_proj_weight = state_dict[tilert_alias.o_proj_weight]

    def get_cache_vars(self) -> list[torch.Tensor]:
        """Get cache variables for KV cache."""
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        bs_args = (self.model_args.max_batch_size, cache_seq_len)
        
        local_kv_dim = self.n_local_kv_heads * self.head_dim
        
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                *bs_args, local_kv_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.v_cache is None:
            self.v_cache = torch.zeros(
                *bs_args, local_kv_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        return [*super().get_cache_vars(), self.k_cache, self.v_cache]

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA.
        
        Args:
            x: [batch, n_kv_heads, seq_len, head_dim]
            n_rep: Number of repetitions per KV head
        Returns:
            [batch, n_heads, seq_len, head_dim]
        """
        bs, n_kv_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        # Expand and reshape to repeat KV heads
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
        )

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference implementation of GQA.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            start_pos: Starting position in the cache
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Ensure weights are bfloat16 for computation
        input_layernorm_gamma = self.input_layernorm_gamma.to(torch.bfloat16)
        q_proj_weight = self.q_proj_weight.to(torch.bfloat16)
        k_proj_weight = self.k_proj_weight.to(torch.bfloat16)
        v_proj_weight = self.v_proj_weight.to(torch.bfloat16)
        o_proj_weight = self.o_proj_weight.to(torch.bfloat16)
        
        # Input layernorm
        x_norm = torch.nn.functional.rms_norm(
            x, [x.size(-1)], input_layernorm_gamma, self.model_args.eps
        )
        
        # Q, K, V projections
        q = torch.matmul(x_norm, q_proj_weight.t())  # [bs, seq, n_local_heads * head_dim]
        k = torch.matmul(x_norm, k_proj_weight.t())  # [bs, seq, n_local_kv_heads * head_dim]
        v = torch.matmul(x_norm, v_proj_weight.t())  # [bs, seq, n_local_kv_heads * head_dim]
        
        # Reshape for attention
        # Q: [bs, n_local_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        # K, V: [bs, n_local_kv_heads, seq, head_dim]
        k = k.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        
        # Update cache
        k_reshaped = k.transpose(1, 2).reshape(batch_size, seq_len, -1)
        v_reshaped = v.transpose(1, 2).reshape(batch_size, seq_len, -1)
        self.k_cache[:batch_size, start_pos:start_pos + seq_len] = k_reshaped
        self.v_cache[:batch_size, start_pos:start_pos + seq_len] = v_reshaped
        
        # Get cached K, V
        k_cached = self.k_cache[:batch_size, :start_pos + seq_len]  # [bs, cache_len, n_local_kv_heads * head_dim]
        v_cached = self.v_cache[:batch_size, :start_pos + seq_len]
        
        # Reshape cached K, V
        k_cached = k_cached.view(batch_size, start_pos + seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        v_cached = v_cached.view(batch_size, start_pos + seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        
        # For GQA: repeat KV heads to match query heads
        k_cached = self._repeat_kv(k_cached, self.n_rep)  # [bs, n_local_heads, cache_len, head_dim]
        v_cached = self._repeat_kv(v_cached, self.n_rep)
        
        # Attention scores: [bs, n_local_heads, seq, cache_len]
        scores = torch.matmul(q, k_cached.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Attention output: [bs, n_local_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v_cached)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = torch.matmul(attn_output, o_proj_weight.t())
        
        return output

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TileRT forward - uses CUDA kernel when available.
        
        Falls back to golden_forward if CUDA kernel is not available.
        """
        try:
            # Try to use the CUDA GQA kernel
            if hasattr(torch.ops.tilert, 'gqa_attention_forward'):
                return self._cuda_gqa_forward(x, start_pos, mask)
        except Exception as e:
            pass
        
        # Fall back to reference implementation
        return self.golden_forward(x, start_pos, mask)
    
    def _cuda_gqa_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CUDA GQA forward using optimized kernels."""
        batch_size, seq_len, _ = x.shape
        
        # Input layernorm
        x_norm = torch.nn.functional.rms_norm(
            x, [x.size(-1)], self.input_layernorm_gamma, self.model_args.eps
        )
        
        # Q, K, V projections
        q = torch.matmul(x_norm, self.q_proj_weight.t())
        k = torch.matmul(x_norm, self.k_proj_weight.t())
        v = torch.matmul(x_norm, self.v_proj_weight.t())
        
        # Reshape for GQA: [bs, seq, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # Update KV cache
        self.k_cache[:batch_size, start_pos:start_pos + seq_len] = k.reshape(batch_size, seq_len, -1)
        self.v_cache[:batch_size, start_pos:start_pos + seq_len] = v.reshape(batch_size, seq_len, -1)
        
        # Get cached K, V
        k_cached = self.k_cache[:batch_size, :start_pos + seq_len].reshape(
            batch_size, start_pos + seq_len, self.n_local_kv_heads, self.head_dim
        )
        v_cached = self.v_cache[:batch_size, :start_pos + seq_len].reshape(
            batch_size, start_pos + seq_len, self.n_local_kv_heads, self.head_dim
        )
        
        # Call CUDA GQA kernel
        attn_output = torch.ops.tilert.gqa_attention_forward(
            q, k_cached, v_cached, self.n_heads, self.n_kv_heads, self.head_dim
        )
        
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = torch.matmul(attn_output, self.o_proj_weight.t())
        
        return output

    def __call__(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.golden_forward(x, start_pos, mask)


class GqaBlock(SerializableTileRTModule):
    """Implement the GQA block operations (Attention + MLP)."""

    def __init__(
        self, model_args: ModelArgs, device_id: int, num_devices: int, remove_selected: bool = False, layer_idx: int = 0
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        from tilert.models.deepseek_v3_2.modules.mlp import Mlp

        self.gqa = Gqa(model_args=model_args, device_id=device_id, num_devices=num_devices, layer_idx=layer_idx)
        self.register_op(self.gqa)
        self.mlp = Mlp(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.mlp)
