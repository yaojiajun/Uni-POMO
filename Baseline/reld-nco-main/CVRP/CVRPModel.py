import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.forcing_first_step = model_params['forcing_first_step']

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        self.node_demand = None

        # Added: Save "Best Start Node Index for each direction"
        # (Index in encoded_nodes; since 0 is depot, we start from 1)
        self.nearest_start_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand

        # 1. Retain basic Encoder encoding
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)

        # 2. Process demand data
        warehouse_demand = torch.zeros_like(node_demand[:, :1])
        self.node_demand = torch.cat((warehouse_demand, node_demand), dim=1)

        self.decoder.set_kv(self.encoded_nodes)

        # Note: self.nearest_start_nodes is no longer calculated here,
        # completely relying on real-time probability output from Decoder.

    def one_step_rollout(self, state, cur_dist, current_node, eval_type='greedy'):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]

        problem_size = state.ninf_mask.size(-1) - 1
        a = 1 / (1 + 2 ** ((100 - problem_size) / 50))
        greedy_count = int(a * 100)
        sample_count = problem_size - greedy_count

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            # Calculate probability distribution for all nodes at current step
            # probs shape: (Batch, multi_width, Problem+1)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(
                encoded_last_node,
                state.load,
                cur_dist,
                self.node_demand,
                current_node,
                ninf_mask=state.ninf_mask,
            )

            # =========================================================================
            # SCENARIO 1: Step 1 (Start Node Selection) - Mixed Strategy: Top-Prob + Sampling
            # =========================================================================
            if state.selected_count == 1:

                # At the start node, all path histories are the same, so distributions are identical
                # Take the first dimension's probability as representative: (Batch, N+1)
                prob_distribution = probs[:, 0, :]

                # --- Part A: Greedy Portion (Top-Prob) ---
                # topk_vals: probability values, topk_indices: action indices
                # shape: (Batch, GREEDY_COUNT)
                topk_vals, topk_indices = torch.topk(prob_distribution, k=greedy_count, dim=1)

                # --- Part B: Sampling Portion ---
                if sample_count > 0:
                    # 1. Sampling Actions
                    # Use multinomial for sampling
                    sampled_indices = prob_distribution.multinomial(sample_count,
                                                                    replacement=True)  # (Batch, SAMPLE_COUNT)

                    # 2. Gather actual probability values for sampled actions
                    sampled_probs = prob_distribution.gather(1, sampled_indices)  # (Batch, SAMPLE_COUNT)
                else:
                    sampled_indices = torch.empty(batch_size, 0, dtype=torch.long, device=device)
                    sampled_probs = torch.empty(batch_size, 0, device=device)

                # --- Part C: Concatenate Actions ---
                selected = torch.cat([topk_indices, sampled_indices], dim=1)

                # --- Part D: Concatenate Probabilities (Return actual probability values) ---
                # Concatenate TopK probabilities and sampled probabilities
                prob = torch.cat([topk_vals, sampled_probs], dim=1)

            # =========================================================================
            # SCENARIO 2: Step > 1 (Subsequent Steps) - Normal Independent Sampling
            # =========================================================================
            else:
                if eval_type == 'sample':
                    with torch.no_grad():
                        # Independent sampling for each path
                        probs_flat = probs.reshape(batch_size * multi_width, -1)
                        selected_flat = probs_flat.multinomial(1).squeeze(1)
                        selected = selected_flat.reshape(batch_size, multi_width)

                    prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                    if not (prob != 0).all():
                        prob += 1e-6
                else:
                    with torch.no_grad():
                        selected = probs.argmax(dim=2).detach()
                    prob = None

        return selected, prob


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)

        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = input1 + multi_head_out
        out2 = self.feed_forward(out1)
        out3 = out1 + out2

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # IDT & FF
        self.capacity_mapping = nn.Linear(1, embedding_dim, bias=False)
        self.feed_forward = FeedForward(**model_params)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)

        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, cur_dist, node_demand, current_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # cur_dist.shape: (batch, pomo, problem)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']
        problem_size = ninf_mask.shape[2]

        # ============================================================
        # 1. Prepare Input Features (Concatenation & Attention)
        # ============================================================

        # Concatenate Last Node Embedding and current Load
        # input_cat shape: (batch, pomo, embedding + 1)
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        # Calculate Query
        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Multi-Head Attention
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # Residual + IDT (Identity) + Capacity Mapping
        mh_atten_out = mh_atten_out + encoded_last_node + self.capacity_mapping(load[:, :, None].clone())

        # Feed Forward + Residual
        q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

        # Single-Head Attention (Calculate baseline Logits)
        score = torch.matmul(q_refined, self.single_head_key)
        # shape: (batch, pomo, problem)

        # Scaling
        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        score_scaled = score / sqrt_embedding_dim

        # ============================================================
        # 2. [Core Modification] Distance Normalization
        # ============================================================

        # Calculate heuristic signal using negative log distance
        # Add 1e-6 to prevent log(0)
        raw_heuristic = -torch.log(cur_dist + 1e-6)
        raw_noise = torch.rand_like(score_scaled)

        N = cur_dist.size(-1)
        a = 1 / (1 + 2 ** ((100 - N) / 50))
        b = 1.0 - a

        # ============================================================
        # 3. Calculate Final Logits (Attention - Log(Dist) + Noise)
        # ============================================================
        mixed_signal = a * raw_heuristic + b * raw_noise
        score = score_scaled + mixed_signal

        # ============================================================
        # 4. Post-processing (Clipping & Masking & Softmax)
        # ============================================================

        logit_clipping = self.model_params['logit_clipping']

        # Tanh Clipping: Prevent exploding gradients from large Logit values
        score_clipped = logit_clipping * torch.tanh(score)

        # Apply Mask (Block visited or unreachable nodes)
        score_masked = score_clipped + ninf_mask

        # Softmax normalization to probability distribution
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    if len(qkv.shape) == 4:
        qkv = qkv.reshape(qkv.size(0) * qkv.size(1), qkv.size(2), qkv.size(3))
        # shape: (batch * multi, n, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))