import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = VRP_Encoder(**model_params)
        self.decoder = VRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_earlyTW = reset_state.node_earlyTW
        node_lateTW = reset_state.node_lateTW

        # --- New: Save Time Window info for Decoder use ---
        # Assume depot time window is [0, 10] or sufficiently large
        depot_TW = torch.tensor([0.0, 10.0], device=node_earlyTW.device).view(1, 1, 2).expand(node_earlyTW.size(0), 1,
                                                                                              2)
        node_TW = torch.cat((node_earlyTW[:, :, None], node_lateTW[:, :, None]), dim=2)  # (batch, problem, 2)

        # Concat into (batch, problem+1, 2), index 0 is depot
        self.time_windows = torch.cat((depot_TW, node_TW), dim=1)
        # ---------------------------------------

        # Maintain original Encoder logic
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        node_xy_demand_TW = torch.cat((node_xy_demand, node_TW), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW)
        self.decoder.set_kv(self.encoded_nodes)
        self.all_nodes_xy = torch.cat((reset_state.depot_xy, reset_state.node_xy), dim=1)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO Start
            # 1. Get coordinates of selected node from previous step
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, pomo_size)
            selected_coords = self.all_nodes_xy[batch_idx, state.current_node]
            # 2. Calculate current distances (Batch, POMO, Problem+1)
            customer_coords = self.all_nodes_xy
            cur_dist = torch.cdist(selected_coords, customer_coords, p=2)
            # 3. Get Embedding of the last node
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            # 4. Obtain probability distribution (Batch, POMO, Problem+1)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, state.time,
                                 ninf_mask=state.ninf_mask, time_windows=self.time_windows)

            if state.selected_count == 1:
                # Set ratio of Greedy and Sampling
                problem_size = state.ninf_mask.size(-1) - 1
                a = 1 / (1 + 2 ** ((100 - problem_size) / 50))
                greedy_count = int(a * 100)
                sample_count = problem_size - greedy_count

                # Extract representative distribution from the first path (as all POMO states are identical at Step 1)
                # prob_distribution shape: (Batch, N)
                prob_distribution = probs[:, 0, :]

                # --- Part A: Greedy Portion (Top-Prob) ---
                topk_vals, topk_indices = torch.topk(prob_distribution, k=greedy_count, dim=1)

                if sample_count > 0:
                    # 1. Sampling Actions
                    # Sample sample_count actions based on the probability distribution
                    selected_sample = prob_distribution.multinomial(sample_count,
                                                                    replacement=True)  # (Batch, sample_count)

                    # 2. Gather Probabilities for sampled actions
                    sample_probs = prob_distribution.gather(1, selected_sample)  # (Batch, sample_count)

                    # --- Part C: Concatenate Actions ---
                    selected = torch.cat([topk_indices, selected_sample], dim=1)

                    # --- Part D: Concatenate Probabilities ---
                    prob = torch.cat([topk_vals, sample_probs], dim=1)

                else:
                    # If sample_count == 0 (All Greedy)
                    selected = topk_indices
                    prob = topk_vals

        else:
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, pomo_size)
            selected_coords = self.all_nodes_xy[batch_idx, state.current_node]
            customer_coords = self.all_nodes_xy
            cur_dist = torch.cdist(selected_coords, customer_coords, p=2)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, state.time,
                                 ninf_mask=state.ninf_mask, time_windows=self.time_windows)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # To fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


########################################
# ENCODER
########################################

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_TW):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_TW)
        # 5 features: x_coord, y_coord, demands, earlyTW, lateTW

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out)

        return out


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

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


########################################
# DECODER
########################################

class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, load, cur_dist, time, ninf_mask, time_windows=None):
        head_num = self.model_params['head_num']

        # Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], time[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        q = q_last
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # Basic Score Calculation
        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim

        # ============================================================
        # [Core Modification] Spatio-Temporal Mixed Cost Injection
        # ============================================================
        # 1. Calculate Wait Time (Temporal)
        current_time = time.unsqueeze(2)  # (batch, pomo, 1)

        if time_windows is not None:
            # Expand to (batch, 1, problem+1) to align with POMO dimension
            starts_exp = time_windows[:, :, 0].unsqueeze(1)
            arrival_time = current_time + cur_dist
            # wait_time = max(0, start_time - arrival_time)
            wait_matrix = (starts_exp - arrival_time).clamp(min=0)
        else:
            wait_matrix = torch.zeros_like(cur_dist)

        # 2. Heuristic Signal Calculation
        st_cost = cur_dist + cur_dist  # Integrated spatio-temporal cost
        raw_heuristic = -torch.log(st_cost + 1e-6)

        # 3. Noise Signal (Ablation: Switchable between Uniform/Gaussian/Exponential)
        raw_noise = torch.rand_like(score_scaled)

        N = cur_dist.size(-1)
        a = 1 / (1 + 2 ** ((100 - N) / 50))
        b = 1.0 - a

        # 4. Mixed Injection
        # -log(st_cost) rewards nodes that are nearby and require no waiting
        mixed_signal = a * raw_heuristic + b * raw_noise

        score = score_scaled + mixed_signal
        # ============================================================

        score_clipped = logit_clipping * torch.tanh(score)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input1, input2):
        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))