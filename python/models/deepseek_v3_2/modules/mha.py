"""MHA (Multi-Head Attention) module for GLM-4.5-Air and similar models."""

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj


class MhaRefWeightsAlias:
    """Reference weights alias for standard MHA."""

    def __init__(self, layer_idx: int = 0):
        self.layer_idx = layer_idx

    def __call__(self) -> list[str]:
        return [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]


class MhaTilertWeightsAlias:
    """TileRT weights alias for standard MHA."""

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


class Mha(SerializableTileRTModule):
    """Implement the standard Multi-Head Attention (MHA) operations.
    
    This is for models like GLM-4.5-Air that use standard MHA instead of MLA.
    """

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int, layer_idx: int = 0):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.n_local_heads = self.n_heads // num_devices
        self.head_dim = self.dim // self.n_heads
        self.layer_idx = layer_idx
        
        # For GLM-4.5-Air: dim=4096, n_heads=32, head_dim=128
        
        self.tilert_weights_alias = MhaTilertWeightsAlias()
        self.ref_weights_alias = MhaRefWeightsAlias()

        self.q_cache: torch.Tensor | None = None
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def device_sharding(
        self, weights_map: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Shard MHA weights across devices.
        
        For standard MHA, we shard Q, K, V, O projections by heads.
        Returns per-device weights dictionary.
        """
        # Find the layer prefix from the weights_map keys
        sample_key = list(weights_map.keys())[0]
        # Extract layer index from key like "model.layers.0.input_layernorm.weight"
        parts = sample_key.split(".")
        layer_idx = parts[2] if len(parts) > 2 else "0"
        prefix = f"model.layers.{layer_idx}."
        
        # Input layernorm - replicated across devices
        input_layernorm_weight = weights_map[f"{prefix}input_layernorm.weight"]
        
        # Q projection: [dim, dim] 
        q_proj_weight_full = weights_map[f"{prefix}self_attn.q_proj.weight"]  # [dim, dim]
        
        # K projection: [dim, dim]
        k_proj_weight_full = weights_map[f"{prefix}self_attn.k_proj.weight"]
        
        # V projection: [dim, dim]
        v_proj_weight_full = weights_map[f"{prefix}self_attn.v_proj.weight"]
        
        # O projection: [dim, dim]
        o_proj_weight_full = weights_map[f"{prefix}self_attn.o_proj.weight"]
        
        # Create sharded weights for all devices
        sharded_weights = {}
        tilert_alias = self.tilert_weights_alias
        
        # Shard for each device
        q_sharded = []
        k_sharded = []
        v_sharded = []
        o_sharded = []
        
        for dev_id in range(self.num_devices):
            # Q projection: shard by output dim (num_heads)
            q_proj = q_proj_weight_full.reshape(self.n_heads, self.head_dim, self.dim)
            q_proj = q_proj[dev_id * self.n_local_heads:(dev_id + 1) * self.n_local_heads]
            q_proj = q_proj.reshape(self.n_local_heads * self.head_dim, self.dim)
            q_sharded.append(q_proj)
            
            # K projection: shard by output dim
            k_proj = k_proj_weight_full.reshape(self.n_heads, self.head_dim, self.dim)
            k_proj = k_proj[dev_id * self.n_local_heads:(dev_id + 1) * self.n_local_heads]
            k_proj = k_proj.reshape(self.n_local_heads * self.head_dim, self.dim)
            k_sharded.append(k_proj)
            
            # V projection: shard by output dim
            v_proj = v_proj_weight_full.reshape(self.n_heads, self.head_dim, self.dim)
            v_proj = v_proj[dev_id * self.n_local_heads:(dev_id + 1) * self.n_local_heads]
            v_proj = v_proj.reshape(self.n_local_heads * self.head_dim, self.dim)
            v_sharded.append(v_proj)
            
            # O projection: shard by input dim (num_heads)
            o_proj = o_proj_weight_full.reshape(self.dim, self.n_heads, self.head_dim)
            o_proj = o_proj[:, dev_id * self.n_local_heads:(dev_id + 1) * self.n_local_heads, :]
            o_proj = o_proj.reshape(self.dim, self.n_local_heads * self.head_dim)
            o_sharded.append(o_proj)
        
        # Stack sharded weights
        q_sharded = torch.stack(q_sharded, dim=0)  # [num_devices, n_local_heads * head_dim, dim]
        k_sharded = torch.stack(k_sharded, dim=0)
        v_sharded = torch.stack(v_sharded, dim=0)
        o_sharded = torch.stack(o_sharded, dim=0)  # [num_devices, dim, n_local_heads * head_dim]
        
        # Input layernorm - replicate across devices
        gamma_sharded = input_layernorm_weight[None, ...].repeat(self.num_devices, 1)
        
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
        
        local_dim = self.n_local_heads * self.head_dim
        
        if self.q_cache is None:
            self.q_cache = torch.zeros(
                *bs_args, local_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                *bs_args, local_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.v_cache is None:
            self.v_cache = torch.zeros(
                *bs_args, local_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        return [*super().get_cache_vars(), self.q_cache, self.k_cache, self.v_cache]

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference implementation of standard MHA.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            start_pos: Starting position in the cache
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input layernorm
        x_norm = torch.nn.functional.rms_norm(
            x.float(), [x.size(-1)], self.input_layernorm_gamma, self.model_args.eps
        ).to(torch.bfloat16)
        
        # Q, K, V projections
        q = torch.matmul(x_norm, self.q_proj_weight.t())  # [bs, seq, n_local_heads * head_dim]
        k = torch.matmul(x_norm, self.k_proj_weight.t())
        v = torch.matmul(x_norm, self.v_proj_weight.t())
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        
        # Update cache
        self.k_cache[:batch_size, start_pos:start_pos + seq_len] = k.transpose(1, 2).reshape(batch_size, seq_len, -1)
        self.v_cache[:batch_size, start_pos:start_pos + seq_len] = v.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Get cached K, V
        k_cached = self.k_cache[:batch_size, :start_pos + seq_len]
        v_cached = self.v_cache[:batch_size, :start_pos + seq_len]
        
        # Reshape cached K, V
        k_cached = k_cached.view(batch_size, start_pos + seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        v_cached = v_cached.view(batch_size, start_pos + seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k_cached.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, v_cached)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = torch.matmul(attn_output, self.o_proj_weight.t())
        
        return output

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TileRT forward - delegates to golden_forward for now.
        
        In the future, this should call the optimized TileRT kernels.
        """
        return self.golden_forward(x, start_pos, mask)

    def __call__(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.golden_forward(x, start_pos, mask)


class MhaBlock(SerializableTileRTModule):
    """Implement the MHA block operations (Attention + MLP)."""

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

        self.mha = Mha(model_args=model_args, device_id=device_id, num_devices=num_devices, layer_idx=layer_idx)
        self.register_op(self.mha)
        self.mlp = Mlp(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.mlp)
