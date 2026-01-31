# -*- coding: utf-8 -*-
"""
Heuristic-Enhanced AttentionModelDecoder (AR, Pointer)
"""

from dataclasses import dataclass, fields
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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

log = get_pylogger(__name__)

# ... (Heuristics import logic remains unchanged) ...

ATTENTION_BIAS_decoder = False
ATTENTION_BIAS_decoder1 = False


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


def _build_heuristics_context(td: TensorDict, env: RL4COEnvBase) -> Optional[Dict[str, Any]]:
    """Builds the context required for heuristic bias calculation."""
    try:
        all_nodes_xy = td.get("coords", None)
        if all_nodes_xy is None:
            all_nodes_xy = td.get("locs", None)

        all_node_demands = td.get("demands", None)
        if all_node_demands is None:
            all_node_demands = td.get("demand", None)

        all_time_windows = td.get("time_windows", None)

        # ... (Context extraction from static/env remains unchanged) ...

        if all_nodes_xy is None or all_node_demands is None:
            log.warning("[Heuristics] Missing 'coords/locs' or 'demands/demand' in td/env; skip heuristic biases.")
            return None

        if all_time_windows is None:
            all_time_windows = torch.zeros(
                all_nodes_xy.size(0), all_nodes_xy.size(1), 2, device=all_nodes_xy.device
            )

        problem_type = problem_type = getattr(env, "problem_type", getattr(env, "name", "VRP"))

        attention_bias2 = None
        if ATTENTION_BIAS_decoder:
            try:
                dist = torch.cdist(all_nodes_xy, all_nodes_xy, p=2)
                out = []
                for i in range(all_nodes_xy.size(0)):
                    if heuristics_decoder is not None:
                        out.append(heuristics_decoder(dist[i], all_node_demands[i], all_time_windows[i]))
                    elif heuristics_basic is not None:
                        out.append(heuristics_basic(dist[i], all_node_demands[i]))
                    else:
                        out.append(torch.zeros_like(all_node_demands[i]))
                attention_bias2 = torch.stack(out, dim=0)
            except Exception as e:
                log.warning(f"[Heuristics] Static decoder bias failed: {e}")
                attention_bias2 = None

        return {
            "all_nodes_xy": all_nodes_xy,
            "all_node_demands": all_node_demands,
            "all_time_windows": all_time_windows,
            "problem_type": problem_type,
            "attention_bias2": attention_bias2,
        }
    except Exception as e:
        log.warning(f"[Heuristics] build context error: {e}")
        return None


class AttentionModelDecoder(AutoregressiveDecoder):
    # ... (__init__ remains unchanged) ...

    def forward(
            self,
            td: TensorDict,
            cached: PrecomputedCache,
            num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        import math
        from einops import rearrange

        # 0. Cache and Dimension Handling
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1
        if has_dyn_emb_multi_start:
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        # 1. Compute Attention Q/K/V
        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        mask = td["action_mask"]

        # Compute baseline Logits (Attention Score)
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # ============================================================
        # [CORE MODIFICATION] Construct Spatiotemporal Cost Matrix
        # with Dynamic Weighting
        # ============================================================

        # --- A. Data Extraction and Dimension Alignment ---
        current_node = td["current_node"]  # (Batch, POMO)
        B, P = current_node.shape

        # Extract node coordinates
        all_nodes_xy = td["locs"]
        if all_nodes_xy.dim() == 4:
            customer_coords = all_nodes_xy[:, 0, ...]  # (Batch, N, 2)
        else:
            customer_coords = all_nodes_xy  # (Batch, N, 2)

        # N: Current problem size
        N = customer_coords.shape[1]

        # Get coordinates of the current node
        batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, P)
        selected_coords = customer_coords[batch_idx, current_node]  # (Batch, POMO, 2)

        # --- B. Compute Dynamic Weighting Coefficient (Sigmoid-based) ---
        # (Weighting coefficient 'a' used below in fusion)

        # --- C. Compute Physical Distance Matrix (Spatial) ---
        dist_matrix = torch.cdist(selected_coords, customer_coords, p=2)  # (Batch, POMO, N)

        # --- D. Compute Wait Time Matrix (Temporal) ---
        current_time = td["current_time"].view(B, P, 1)

        all_earliest_starts = td["time_windows"][..., 0]
        if all_earliest_starts.dim() == 2:
            starts_exp = all_earliest_starts.unsqueeze(1)
        else:
            starts_exp = all_earliest_starts

        # Expected arrival time = Current time + Distance
        arrival_time_matrix = current_time + dist_matrix
        # Wait time = max(0, start_time - arrival_time)
        wait_matrix = (starts_exp - arrival_time_matrix).clamp(min=0)

        # --- E. Normalization and Fusion ---
        def normalize_matrix(mat):
            min_v = mat.min(dim=-1, keepdim=True).values
            max_v = mat.max(dim=-1, keepdim=True).values
            return (mat - min_v) / (max_v - min_v + 1e-6)

        norm_dist = normalize_matrix(dist_matrix)
        norm_wait = normalize_matrix(wait_matrix)

        # Fuse Spatiotemporal costs: Consider both travel distance and time window waiting
        cur_dist = dist_matrix + wait_matrix

        # --- F. Generate Dynamic Weighted Noise Injection ---
        base_noise = torch.rand_like(cur_dist)
        problem_size = customer_coords.shape[1]

        # Sigmoid-like scaling factor based on problem size N
        a = 1 / (1 + 2 ** ((100 - problem_size) / 50))

        # Inject combined heuristic signal and noise into logits
        # -log(cur_dist) rewards closer nodes with shorter wait times
        logits = logits - a * torch.log(cur_dist + 1e-6) + base_noise * (1 - a)

        # 3. Restore dimensions (specifically for multi-start decoding)
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        return logits, mask