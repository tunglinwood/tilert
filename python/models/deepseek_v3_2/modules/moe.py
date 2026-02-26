from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.expert_down_allreduce import (
    ExpertDownAllReduce,
    ExpertDownAllReduceAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_expert_proj import RMSNormExpertProj


class Moe(SerializableTileRTModule):
    """Implement the MOE operations."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_expert_proj = RMSNormExpertProj(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_expert_proj)

        self.exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_5":
            # GLM-5 uses FP16MMA (dim=6144)
            self.exp_sel_up_gate_silu.algorithm = ExpertSelectUpGateSiLUAlgorithm.FP16MMA
        elif model_args.arch_name == "glm_4_5_air":
            # GLM-4.5-Air uses BF16 pass-through (dim=4096, different expert dims)
            self.exp_sel_up_gate_silu.algorithm = ExpertSelectUpGateSiLUAlgorithm.BF16
        self.register_op(self.exp_sel_up_gate_silu)
        self.expert_down_allreduce = ExpertDownAllReduce(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_4_5_air":
            # GLM-4.5-Air uses BF16 pass-through
            self.expert_down_allreduce.algorithm = ExpertDownAllReduceAlgorithm.BF16
        self.register_op(self.expert_down_allreduce)


class MoeBlock(SerializableTileRTModule):
    """Implement the MOE block operations."""

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
        self.moe = Moe(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.moe)
