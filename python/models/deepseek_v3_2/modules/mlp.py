from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.down_allreduce import (
    DownAllReduce,
    DownAllReduceAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)


class Mlp(SerializableTileRTModule):
    """Implement the MLP operations."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_mlp_up_gate_silu = RMSNormUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        if model_args.arch_name == "glm_5":
            # GLM-5 uses FP16MMA (dim=6144)
            self.rmsnorm_mlp_up_gate_silu.algorithm = RMSNormUpGateSiLUAlgorithm.FP16MMA
        elif model_args.arch_name == "glm_4_5_air":
            # GLM-4.5-Air uses BF16 pass-through (dim=4096, different intermediate dims)
            self.rmsnorm_mlp_up_gate_silu.algorithm = RMSNormUpGateSiLUAlgorithm.BF16
        self.register_op(self.rmsnorm_mlp_up_gate_silu)
        self.rmsnorm_mlp_down = DownAllReduce(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_4_5_air":
            # GLM-4.5-Air uses BF16 pass-through
            self.rmsnorm_mlp_down.algorithm = DownAllReduceAlgorithm.BF16
        self.register_op(self.rmsnorm_mlp_down)


class MlpBlock(SerializableTileRTModule):
    """Implement the MLP block operations with attention."""

    def __init__(
        self, model_args: ModelArgs, device_id: int, num_devices: int, remove_selected: bool = False, layer_idx: int = 0
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        # Use GQA for GLM-4.5-Air (n_kv_heads < n_heads), MLA for others
        n_kv_heads = getattr(model_args, 'n_kv_heads', model_args.n_heads)
        if n_kv_heads < model_args.n_heads:
            # GQA (GLM-4.5-Air style)
            from tilert.models.deepseek_v3_2.modules.gqa import Gqa
            self.attn = Gqa(model_args=model_args, device_id=device_id, num_devices=num_devices, layer_idx=layer_idx)
        elif getattr(model_args, 'q_lora_rank', 0) == 0:
            # Standard MHA
            from tilert.models.deepseek_v3_2.modules.mha import Mha
            self.attn = Mha(model_args=model_args, device_id=device_id, num_devices=num_devices, layer_idx=layer_idx)
        else:
            # MLA (DeepSeek-V3.2 / GLM-5 style)
            from tilert.models.deepseek_v3_2.modules.mla import Mla
            self.attn = Mla(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.attn)
        self.mlp = Mlp(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.mlp)
