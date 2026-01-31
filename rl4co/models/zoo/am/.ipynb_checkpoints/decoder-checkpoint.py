# -*- coding: utf-8 -*-
"""
Heuristic-Enhanced AttentionModelDecoder (AR, Pointer)
- 启发式开关完全由全局常量控制： ATTENTION_BIAS_encoder / ATTENTION_BIAS_decoder / ATTENTION_BIAS_decoder1
- 偏置命名：attention_bias2（静态一次性）、attention_bias3（动态逐步）
- 与 MTVRPEnv 的 td 字段完全对齐
- 修正：移除 ctx，直接从 td 解析静态和动态数据，避免预构建 ctx 以简化逻辑。
  动态偏置 (bias3) 在 forward 中直接从 td 提取所有必要参数，送入 heuristics。
  heuristics 参数严格匹配要求：current_distance_matrix (P, N+1), delivery_node_demands (N+1), load (P),
  delivery_node_demands_open (N+1), load_open (P), all_time_windows (N+1, 2), estimated_arrival (P, N+1),
  pickup_node_demands (N+1), cur_len (P)。
  假设 current_route_length 是 remaining length budget；delivery_node_demands_open 和 load_open 对于非 OVRP 使用 zeros。
- 维度修正：处理 all_nodes_xy 等静态字段可能带有 POMO 维度 (B, P, N+1, 2) 的情况，挤压或扩展以确保一致性。
- 修改：移除静态偏置存储 (_bias2_full) 及相关逻辑（包括 _build_static_bias、pre_decoder_hook 中的静态部分、forward 中的 bias2），因为用户指示“静态偏置存储不要”。
- 调整 heuristics 调用为指定结构，参数名称匹配（load=current_load, load_open=current_load_open, estimated_arrival=arrival_times, cur_len=current_length）。
"""
import os, sys
# 计算到包含 “Hercules/” 的目录：从 .../Hercules/problems/mt_routefinder_unified/eval.py 往上三级
_PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# _PROJECT_PARENT 现在应该是 /root/autodl-tmp/yao
if _PROJECT_PARENT not in sys.path:
    sys.path.insert(0, _PROJECT_PARENT)

# 如果 Hercules 目录没有 __init__.py，又不想改文件，可以动态注册成命名空间包（兜底）
import importlib.util, types
_hercules_dir = os.path.join(_PROJECT_PARENT, "Hercules")
if "Hercules" not in sys.modules and os.path.isdir(_hercules_dir):
    spec = importlib.util.spec_from_loader("Hercules", loader=None, origin=_hercules_dir)
    hercules_pkg = importlib.util.module_from_spec(spec)
    hercules_pkg.__path__ = [_hercules_dir]  # 命名空间包
    sys.modules["Hercules"] = hercules_pkg

from dataclasses import dataclass, fields
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor
from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import batchify, unbatchify
# from Hercules.problems.mt_routefinder_unified.gpt import heuristics_v2 as heuristics

log = get_pylogger(__name__)

# ========= 启发式导入 =========
# from routefinder.models.gpt_decoder import heuristics_decoder
# from routefinder.models.gpt_basic import basic_score_matrix as heuristics_basic
# ==============================================

# ========= 全局开关 =========
ATTENTION_BIAS_encoder = False  # 编码器静态距离偏置（此文件不使用）
ATTENTION_BIAS_decoder = False  # 解码器静态启发式偏置（一次性，(B,N+1,N+1)）- 已移除相关代码
ATTENTION_BIAS_decoder1 = False  # 解码器动态启发式偏置（逐步，(B,P,N+1)）
# ==============================================

# ---------- 小工具 ----------
def _td_get(td: TensorDict, *keys, default=None):
    for k in keys:
        if k in td.keys():
            return td[k]
    return default

# ---------- PrecomputedCache ----------
@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Tensor | float
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)

class AttentionModelDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
        heuristic_scale: float = 1.0,
    ):
        super().__init__()
        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.heuristic_scale = heuristic_scale
        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = not isinstance(self.dynamic_embedding, StaticEmbedding)
        if pointer is None:
            pointer_attn_class = PointerAttention if moe_kwargs is None else PointerAttnMoE
            pointer = pointer_attn_class(
                embed_dim, num_heads, mask_inner=mask_inner, out_bias=out_bias_pointer_attn,
                check_nan=check_nan, sdpa_fn=sdpa_fn, moe_kwargs=moe_kwargs
            )
        self.pointer = pointer
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=linear_bias)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context
        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)
        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        gk_s, gv_s, lk_s = cached.glimpse_key, cached.glimpse_val, cached.logit_key
        gk_d, gv_d, lk_d = self.dynamic_embedding(td)
        return gk_s + gk_d, gv_s + gv_d, lk_s + lk_d

    def pre_decoder_hook(self, td, env, embeddings, num_starts: int = 0):
        cache = self._precompute_cache(embeddings, num_starts=num_starts)
        return td, env, cache

    def _precompute_cache(self, embeddings: torch.Tensor, num_starts: int = 0) -> PrecomputedCache:
        gk, gv, lk = self.project_node_embeddings(embeddings).chunk(3, dim=-1)
        graph_context = self.project_fixed_context(embeddings.mean(1)) if self.use_graph_context else 0
        return PrecomputedCache(embeddings, graph_context, gk, gv, lk)

    import torch
    from typing import Optional
    from tensordict import TensorDict

    import torch
    from typing import Optional
    from tensordict import TensorDict

    import torch
    from typing import Optional
    from tensordict import TensorDict

    import torch
    from typing import Optional
    from tensordict import TensorDict

    import torch
    from typing import Optional
    from tensordict import TensorDict

    def _compute_dynamic_bias(self, td: TensorDict) -> Optional[torch.Tensor]:
        """
        固定维度版本（静态先去掉POMO维）：
          - 静态节点张量:  time_windows/demands/open_route/distance_limit 先压掉 POMO 维
          - 动态量:       locs/current_time/current_route_length/vehicle & used capacities 保留 P 维
          - 统一 device / dtype
          - 返回 (B, P, N+1)
        形状约定：
          locs:                 (B, P, N+1, 2)
          current_node:         (B, P)
          current_time:         (B, P, 1) 或 (B, P)  -> 用作 (B,P,1)
          current_route_length: (B, P, 1) 或 (B, P)  -> 用作 (B,P)
          vehicle_capacity:     (B, P) 或 (B, P, 1)  -> 用作 (B,P)
          used_capacity_*:      (B, P) 或 (B, P, 1)  -> 用作 (B,P)
          time_windows:         (B, P, N+1, 2)       -> 取 [:,0] 得 (B, N+1, 2)
          demand_linehaul:      (B, P, N+1) 或 (B, N+1) -> 取 [:,0] 得 (B, N+1)
          demand_backhaul:      (B, P, N+1) 或 (B, N+1) -> 取 [:,0] 得 (B, N+1)
          open_route:           (B, N+1, 1)          -> squeeze(-1) 得 (B, N+1) bool 掩码
          distance_limit:       (B, N+1, 1)          -> squeeze(-1) 得 (B, N+1)
        """
        device, dtype = td["locs"].device, td["locs"].dtype

        # -------- 动态量（保留 P 维，并统一 device/dtype）--------
        current_node = td["current_node"].to(device=device)  # (B,P)

        B, P = current_node.shape
        all_nodes_xy = td["locs"].to(device=device, dtype=dtype)  # (B,P,N+1,2)
        customer_coords = all_nodes_xy[:, 0, ...]
        _, _, Np1, _ = all_nodes_xy.shape
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, P)
        selected_coords = customer_coords[batch_idx, current_node]
        current_distance_matrix = torch.cdist(selected_coords, customer_coords, p=2)


        time_t = td["current_time"].to(device=device, dtype=dtype)  # (B,P,1) 或 (B,P)
        if time_t.dim() == 2:
            time_t = time_t.unsqueeze(-1)  # -> (B,P,1)

        cur_len_raw = td["current_route_length"].to(device=device, dtype=dtype)  # (B,P,1) 或 (B,P)
        if cur_len_raw.dim() == 3 and cur_len_raw.size(-1) == 1:
            cur_len_raw = cur_len_raw.squeeze(-1)  # -> (B,P)

        vehicle_cap = td["vehicle_capacity"].to(device=device, dtype=dtype)  # (B,P) 或 (B,P,1)
        if vehicle_cap.dim() == 3 and vehicle_cap.size(-1) == 1:
            vehicle_cap = vehicle_cap.squeeze(-1)  # -> (B,P)

        uc_l = td["used_capacity_linehaul"].to(device=device, dtype=dtype)  # (B,P) 或 (B,P,1)
        uc_b = td["used_capacity_backhaul"].to(device=device, dtype=dtype)  # (B,P) 或 (B,P,1)
        if uc_l.dim() == 3 and uc_l.size(-1) == 1: uc_l = uc_l.squeeze(-1)  # -> (B,P)
        if uc_b.dim() == 3 and uc_b.size(-1) == 1: uc_b = uc_b.squeeze(-1)  # -> (B,P)

        load = vehicle_cap - (uc_l + uc_b)  # (B,P)

        # -------- 静态/节点级张量：先去掉 POMO 维，并统一 device/dtype --------
        all_time_windows = td["time_windows"][:, 0].to(device=device, dtype=dtype)  # (B,N+1,2)

        dem_l = td["demand_linehaul"]
        dem_l = dem_l[:, 0] if dem_l.dim() == 3 else dem_l  # (B,N+1)
        dem_l = dem_l.to(device=device, dtype=dtype)

        dem_b = td["demand_backhaul"]
        dem_b = dem_b[:, 0] if dem_b.dim() == 3 else dem_b  # (B,N+1)
        dem_b = dem_b.to(device=device, dtype=dtype)

        or_mask = td["open_route"][:, :, 0].to(device=device).bool()  # (B,N+1)
        distance_limit_nodes = td["distance_limit"][:, :, 0].to(device=device, dtype=dtype)  # (B,N+1)

        # -------- 距离矩阵 & ETA（保留 P 维）--------
        # 当前位置 (B,P,2)
        cur_xy = all_nodes_xy.gather(  # (B,P,1,2)->(B,P,2)
            2, current_node.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, 2)
        ).squeeze(2)
        all_nodes_xy_BP = all_nodes_xy[:, 0, ...]
        # 距离矩阵 (B,P,N+1)
        # current_distance_matrix = torch.cdist(cur_xy, all_nodes_xy_BP, p=2) # (B*P, N+1)
        # current_distance_matrix = cd.view(B, P, Np1)

        # ETA (B,P,N+1)
        estimated_arrival = time_t + current_distance_matrix  # (B,P,1)+(B,P,N+1) -> (B,P,N+1)

        # -------- 送/取拆分（节点级）--------
        pos_c = torch.tensor(2.0, device=device, dtype=dem_l.dtype)
        neg_c = torch.tensor(-2.0, device=device, dtype=dem_l.dtype)
        delivery_node_demands = torch.where(dem_l > 0, dem_l, pos_c)  # (B,N+1)
        pickup_node_demands = torch.where(dem_b > 0, -dem_b, neg_c)  # (B,N+1)

        # 最后 20 个节点 backhaul 和为 0 → 该 batch 取货清零
        batch_tail_zero = (dem_b[..., -20:].sum(dim=-1) == 0)  # (B,)
        if batch_tail_zero.any():
            pickup_node_demands[batch_tail_zero] = 0

        # -------- open-route 三种情况（节点级掩码）--------
        delivery_node_demands_open = torch.zeros_like(delivery_node_demands)  # (B,N+1)
        load_open = torch.zeros_like(load)  # (B,P)
        if or_mask.all():
            delivery_node_demands_open = delivery_node_demands
            load_open = load
        elif or_mask.any():
            delivery_node_demands_open = delivery_node_demands * or_mask.to(device=device,
                                                                            dtype=delivery_node_demands.dtype)
            any_open = or_mask.any(dim=-1)  # (B,)
            load_open[any_open] = load[any_open]
        # 全 False 保持 0

        # -------- 距离上限 → 剩余长度（节点级上限压成每 batch 标量）--------
        # distance_limit_nodes: (B,N+1) → 每 batch 上限 (B,1)
        max_len_per_batch = distance_limit_nodes.max(dim=1, keepdim=True).values  # (B,1)
        # 剩余长度 = 上限 - 已走 (B,1)-(B,P) → 广播成 (B,P)
        cur_len = max_len_per_batch - cur_len_raw  # (B,P)

        # -------- 若所有 late 均为 inf 或 0，则 TW/ETA 置零 --------
        late = all_time_windows[..., 1]  # (B,N+1)
        if torch.isinf(late).all() or (late == 0).all():
            all_time_windows = torch.zeros_like(all_time_windows)  # (B,N+1,2)
            estimated_arrival = torch.zeros_like(estimated_arrival)  # (B,P,N+1)

        # -------- 调 heuristics（逐 batch）--------
        attention_bias_dynamic = torch.stack([
            heuristics(
                current_distance_matrix[i],  # (P, N+1)
                delivery_node_demands[i],  # (N+1)
                load[i],  # (P)
                delivery_node_demands_open[i],  # (N+1)
                load_open[i],  # (P)
                all_time_windows[i],  # (N+1, 2)
                estimated_arrival[i],  # (P, N+1)
                pickup_node_demands[i],  # (N+1)
                cur_len[i],  # (P)
            ) for i in range(B)
        ], dim=0)  # (B,P,N+1)

        # 数值清理
        attention_bias_dynamic = torch.nan_to_num(attention_bias_dynamic, neginf=-float('inf'))
        return attention_bias_dynamic

    def _squeeze_static(self, field: Tensor, P: int) -> Tensor:
        """挤压静态字段的 P 维，取第一个，并挤压尾部单维。"""
        while field.dim() > 0 and field.size(-1) == 1:
            field = field.squeeze(-1)
        if field.dim() == 3 and field.shape[1] == P:
            field = field[:, 0]  # (B, N1, ...)
        return field

    def _expand_dynamic(self, field: Tensor, P: int) -> Tensor:
        """扩展动态字段到 P，如果是 1 则 expand。"""
        if field.size(1) == 1:
            field = field.expand(-1, P)
        return field

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor, Tensor]:
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1
        if has_dyn_emb_multi_start:
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            td = unbatchify(td, num_starts)
        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        current_node: Tensor = _td_get(td, "current_node")
        if current_node is not None:
            if current_node.dim() == 1:
                current_node = current_node.unsqueeze(1)
            B, P = current_node.shape

            # 动态偏置 (bias3)：直接从 td 计算
            bias3 = self._compute_dynamic_bias(td) if ATTENTION_BIAS_decoder1 else None

            # 应用偏置（仅动态）
            if bias3 is not None:
                logits = logits + bias3

        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask