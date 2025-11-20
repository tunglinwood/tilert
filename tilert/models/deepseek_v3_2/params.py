from abc import abstractmethod

import torch

__all__ = [
    "IntermediateMapper",
    "BaseParams",
    "MlaParams",
    "MLPParams",
    "MoEParams",
    "TempVars",
    "DenseLayerParamsKeys",
    "MoELayerParamsKeys",
]


DenseLayerParamsKeys = [
    # MLA params
    "x_rmsnorm_gamma",  # 0
    "qkv_wa_weights",  # 1
    "qkv_wa_scales",  # 2
    "k_weights",  # 3
    "k_bias",  # 4
    "q_rmsnorm_gamma",  # 5
    "q_wb_weights",  # 6
    "q_wb_scales",  # 7
    "id_score_weights",  # 8
    "wkv_b1_weights",  # 9
    "wkv_b1_scales",  # 10
    "kv_rmsnorm_gamma",  # 11
    "wkv_b2_weights",  # 12
    "wkv_b2_scales",  # 13
    "unproj_weights",  # 14
    "unproj_scales",  # 15
    # MLP params
    "unproj_o_gamma",  # 16
    "upgate_weights",  # 17
    "upgate_scales",  # 18
    "down_weights",  # 19
    "down_scales",  # 20
]

MoELayerParamsKeys = [
    # MLA params
    "x_rmsnorm_gamma",  # 0
    "qkv_wa_weights",  # 1
    "qkv_wa_scales",  # 2
    "k_weights",  # 3
    "k_bias",  # 4
    "q_rmsnorm_gamma",  # 5
    "q_wb_weights",  # 6
    "q_wb_scales",  # 7
    "id_score_weights",  # 8
    "wkv_b1_weights",  # 9
    "wkv_b1_scales",  # 10
    "kv_rmsnorm_gamma",  # 11
    "wkv_b2_weights",  # 12
    "wkv_b2_scales",  # 13
    "unproj_weights",  # 14
    "unproj_scales",  # 15
    # MoE params
    "unproj_o_gamma",  # 16
    "exp_proj_weights",  # 17
    "exp_bias",  # 18
    "exp_upgate_weights",  # 19
    "exp_upgate_scales",  # 20
    "exp_down_weights",  # 21
    "exp_down_scales",  # 22
]


class IntermediateMapper:
    """Map the intermediate tensors to the corresponding variables."""

    def __init__(self, intermediate_list: list[torch.Tensor]):
        self.q = intermediate_list[0]
        self.kv = intermediate_list[1]
        self.ki = intermediate_list[2]
        self.q_nope_down = intermediate_list[3]
        self.q_pe = intermediate_list[4]
        self.iq = intermediate_list[5]
        self.iq_rt = intermediate_list[6]
        self.idx_score = intermediate_list[7]
        self.idx_logits = intermediate_list[8]
        self.idx_sels = intermediate_list[9]
        self.q_nope = intermediate_list[10]
        self.o = intermediate_list[11]
        self.o_acc = intermediate_list[12]
        self.o_lse = intermediate_list[13]
        self.o_lse_acc = intermediate_list[14]
        self.proj_o = intermediate_list[15]
        self.unproj_o = intermediate_list[16]
        self.scores = intermediate_list[17]
        self.x_mlp_in = intermediate_list[18]
        self.exp_up_gate = intermediate_list[19]
        self.sel_probs = intermediate_list[20]
        self.sel_indices = intermediate_list[21]
        self.exp_out = intermediate_list[22]
        self.x_rmsnorm = intermediate_list[23]
        self.logits_out = intermediate_list[24]
        self.token_out = intermediate_list[25]


class BaseParams:
    def __init__(self) -> None:
        self._params: list[torch.Tensor] = []

    def register_params(self, param: torch.Tensor) -> torch.Tensor:
        self._params.append(param)
        return param

    def get_params(self) -> list[torch.Tensor]:
        return self._params

    @staticmethod
    @abstractmethod
    def num_params() -> int:
        raise NotImplementedError("Subclasses must implement this method")


class MlaParams(BaseParams):
    def __init__(
        self,
        x_rmsnorm_gamma: torch.Tensor,
        qkv_wa_weights: torch.Tensor,
        qkv_wa_scales: torch.Tensor,
        k_weights: torch.Tensor,
        k_bias: torch.Tensor,
        q_rmsnorm_gamma: torch.Tensor,
        q_wb_weights: torch.Tensor,
        q_wb_scales: torch.Tensor,
        id_score_weights: torch.Tensor,
        wkv_b1_weights: torch.Tensor,
        wkv_b1_scales: torch.Tensor,
        kv_rmsnorm_gamma: torch.Tensor,
        wkv_b2_weights: torch.Tensor,
        wkv_b2_scales: torch.Tensor,
        unproj_weights: torch.Tensor,
        unproj_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.x_rmsnorm_gamma = self.register_params(x_rmsnorm_gamma)
        self.qkv_wa_weights = self.register_params(qkv_wa_weights)
        self.qkv_wa_scales = self.register_params(qkv_wa_scales)
        self.k_weights = self.register_params(k_weights)
        self.k_bias = self.register_params(k_bias)
        self.q_rmsnorm_gamma = self.register_params(q_rmsnorm_gamma)
        self.q_wb_weights = self.register_params(q_wb_weights)
        self.q_wb_scales = self.register_params(q_wb_scales)
        self.id_score_weights = self.register_params(id_score_weights)
        self.wkv_b1_weights = self.register_params(wkv_b1_weights)
        self.wkv_b1_scales = self.register_params(wkv_b1_scales)
        self.kv_rmsnorm_gamma = self.register_params(kv_rmsnorm_gamma)
        self.wkv_b2_weights = self.register_params(wkv_b2_weights)
        self.wkv_b2_scales = self.register_params(wkv_b2_scales)
        self.unproj_weights = self.register_params(unproj_weights)
        self.unproj_scales = self.register_params(unproj_scales)

    @staticmethod
    def num_params() -> int:
        return 16


class MLPParams(BaseParams):
    def __init__(
        self,
        unproj_o_gamma: torch.Tensor,
        upgate_weights: torch.Tensor,
        upgate_scales: torch.Tensor,
        down_weights: torch.Tensor,
        down_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unproj_o_gamma = self.register_params(unproj_o_gamma)
        self.upgate_weights = self.register_params(upgate_weights)
        self.upgate_scales = self.register_params(upgate_scales)
        self.down_weights = self.register_params(down_weights)
        self.down_scales = self.register_params(down_scales)

    @staticmethod
    def num_params() -> int:
        return 5


class MoEParams(BaseParams):
    def __init__(
        self,
        unproj_o_gamma: torch.Tensor,
        exp_proj_weights: torch.Tensor,
        exp_bias: torch.Tensor,
        exp_upgate_weights: torch.Tensor,
        exp_upgate_scales: torch.Tensor,
        exp_down_weights: torch.Tensor,
        exp_down_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unproj_o_gamma = self.register_params(unproj_o_gamma)
        self.exp_proj_weights = self.register_params(exp_proj_weights)
        self.exp_bias = self.register_params(exp_bias)
        self.exp_upgate_weights = self.register_params(exp_upgate_weights)
        self.exp_upgate_scales = self.register_params(exp_upgate_scales)
        self.exp_down_weights = self.register_params(exp_down_weights)
        self.exp_down_scales = self.register_params(exp_down_scales)

    @staticmethod
    def num_params() -> int:
        return 7


class TempVars(BaseParams):
    def __init__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        ki: torch.Tensor,
        q_nope_down: torch.Tensor,
        q_pe: torch.Tensor,
        iq: torch.Tensor,
        iq_rt: torch.Tensor,
        idx_score: torch.Tensor,
        idx_logits: torch.Tensor,
        idx_sels: torch.Tensor,
        q_nope: torch.Tensor,
        o: torch.Tensor,
        o_acc: torch.Tensor,
        o_lse: torch.Tensor,
        o_lse_acc: torch.Tensor,
        proj_o: torch.Tensor,
        unproj_o: torch.Tensor,
        scores: torch.Tensor,
        x_mlp_in: torch.Tensor,
        exp_up_gate: torch.Tensor,
        sel_probs: torch.Tensor,
        sel_indices: torch.Tensor,
        exp_out: torch.Tensor,
        x_rmsnorm: torch.Tensor,
        logits_out: torch.Tensor,
        token_out: torch.Tensor,
    ) -> None:
        super().__init__()
        self.q = self.register_params(q)
        self.kv = self.register_params(kv)
        self.ki = self.register_params(ki)
        self.q_nope_down = self.register_params(q_nope_down)
        self.q_pe = self.register_params(q_pe)
        self.iq = self.register_params(iq)
        self.iq_rt = self.register_params(iq_rt)
        self.idx_score = self.register_params(idx_score)
        self.idx_logits = self.register_params(idx_logits)
        self.idx_sels = self.register_params(idx_sels)
        self.q_nope = self.register_params(q_nope)
        self.o = self.register_params(o)
        self.o_acc = self.register_params(o_acc)
        self.o_lse = self.register_params(o_lse)
        self.o_lse_acc = self.register_params(o_lse_acc)
        self.proj_o = self.register_params(proj_o)
        self.unproj_o = self.register_params(unproj_o)
        self.scores = self.register_params(scores)
        self.x_mlp_in = self.register_params(x_mlp_in)
        self.exp_up_gate = self.register_params(exp_up_gate)
        self.sel_probs = self.register_params(sel_probs)
        self.sel_indices = self.register_params(sel_indices)
        self.exp_out = self.register_params(exp_out)
        self.x_rmsnorm = self.register_params(x_rmsnorm)
        self.logits_out = self.register_params(logits_out)
        self.token_out = self.register_params(token_out)

    @staticmethod
    def num_params() -> int:
        return 26


class CacheVars(BaseParams):
    def __init__(
        self,
        k_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
    ) -> None:
        super().__init__()
        self.k_cache = self.register_params(k_cache)
        self.kv_cache = self.register_params(kv_cache)
        self.pe_cache = self.register_params(pe_cache)

    @staticmethod
    def num_params() -> int:
        return 3


class LLMHeadParams(BaseParams):
    """LLM Head Parameters"""

    def __init__(
        self,
        hidden_rms_gamma: torch.Tensor,
        head_proj_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.hidden_rms_gamma = self.register_params(hidden_rms_gamma)
        self.head_proj_weights = self.register_params(head_proj_weights)

    @staticmethod
    def num_params() -> int:
        return 2
