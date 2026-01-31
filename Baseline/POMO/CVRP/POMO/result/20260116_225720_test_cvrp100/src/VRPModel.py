
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
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)
        all_nodes_xy = torch.cat((reset_state.depot_xy, reset_state.node_xy), dim=1)
        self.all_nodes_xy = all_nodes_xy

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        # ==========================================================
        # 第一步：回到仓库 (First Move)
        # ==========================================================
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        # ==========================================================
        # 第二步：初始节点选择 (Second Move, POMO Start)
        # ==========================================================
        elif state.selected_count == 1:
            # 1. 准备 Decoder 需要的输入
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, pomo_size)
            selected_coords = self.all_nodes_xy[batch_idx, state.current_node]
            customer_coords = self.all_nodes_xy
            cur_dist = torch.cdist(selected_coords, customer_coords, p=2)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            # 2. 获得当前概率分布 (Batch, POMO, Problem+1)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, ninf_mask=state.ninf_mask)

            # --- 核心逻辑：基于问题规模动态扩展 ---

            # 获取问题规模 N (总节点数 - 1个Depot)
            problem_size = state.ninf_mask.size(-1) - 1

            # 设定基础的 Greedy 和 Sampling 数量 (各占问题规模的一半)
            base_greedy_count = problem_size // 2
            base_sample_count = problem_size - base_greedy_count

            # 计算需要整体重复的倍数
            aug = pomo_size // problem_size

            # 提取第一个 POMO 索引的分布作为代表 (初始状态所有 POMO 是一样的)
            prob_distribution = probs[:, 0, :]  # shape: (Batch, Problem+1)

            # --- A. 基础贪婪部分 (Top-K) ---
            topk_vals_base, topk_indices_base = torch.topk(prob_distribution, k=base_greedy_count, dim=1)
            # 块状重复: [1,2..50, 1,2..50...]
            selected_greedy = topk_indices_base.repeat(1, aug)
            prob_greedy = topk_vals_base.repeat(1, aug)

            # --- B. 基础采样部分 ---
            if base_sample_count > 0:
                # 采样 base_sample_count 个动作
                selected_sample_base = prob_distribution.multinomial(base_sample_count, replacement=True)
                sample_probs_base = prob_distribution.gather(1, selected_sample_base)

                # 块状重复采样部分
                selected_sample = selected_sample_base.repeat(1, aug)
                prob_sample = sample_probs_base.repeat(1, aug)

                # 拼接 Greedy 和 Sample
                selected = torch.cat([selected_greedy, selected_sample], dim=1)
                prob = torch.cat([prob_greedy, prob_sample], dim=1)
            else:
                selected = selected_greedy
                prob = prob_greedy

            # --- C. 补齐余数 (针对 pomo_size 不能被 problem_size 整除的情况) ---
            remaining = pomo_size - selected.size(1)
            if remaining > 0:
                selected = torch.cat([selected, selected[:, :remaining]], dim=1)
                prob = torch.cat([prob, prob[:, :remaining]], dim=1)

        # ==========================================================
        # 第三步及以后：常规解码 (Subsequent Moves)
        # ==========================================================
        else:
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, pomo_size)
            selected_coords = self.all_nodes_xy[batch_idx, state.current_node]
            customer_coords = self.all_nodes_xy
            cur_dist = torch.cdist(selected_coords, customer_coords, p=2)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            probs = self.decoder(encoded_last_node, state.load, cur_dist, ninf_mask=state.ninf_mask)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # 修复概率为 0 时的 multinomial bug
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                # 测试模式下通常直接取最大概率
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


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
        self.embedding_node = nn.Linear(3, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # input shape: (batch, problem, 3)
        # 3 features are: x_coord, y_coord, demands
        # embedded_node shape: (batch, problem, embedding)

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

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

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

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


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

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, cur_dist, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        q = q_last
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim

        # ============================================================
        # [核心修改] 自适应平衡门控注入 (Adaptive Injection)
        # ============================================================

        # A. Min-Max 归一化 (保留用户逻辑)
        # shape: (batch, pomo, 1)
        d_min = cur_dist.min(dim=-1, keepdim=True).values
        d_max = cur_dist.max(dim=-1, keepdim=True).values

        # 归一化到 [0, 1]
        norm_dist = (cur_dist - d_min) / (d_max - d_min + 1e-6)

        # B. 准备信号
        # 物理信号: 距离越近 -> norm_dist越小 -> -log越大 (奖励)
        raw_heuristic = -torch.log(norm_dist + 1e-6)

        # 噪音信号: 标准正态分布
        raw_noise = torch.rand_like(score_scaled)

        # C. 计算互补权重 (a + b = 1)
        # 必须使用 sigmoid 保证 a 在 [0,1] 区间
        a = 0.5
        b = 0.5
        # print(a)
        # D. 混合
        # a * 物理 + b * 随机
        mixed_signal = a * raw_heuristic + b * raw_noise
        # mixed_signal = b * raw_noise
        # E. 注入
        # 直接叠加到 score 上
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
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
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


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

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
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))