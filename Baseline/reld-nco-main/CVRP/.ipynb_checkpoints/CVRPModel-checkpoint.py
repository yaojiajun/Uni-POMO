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

        # 新增：保存“各方向最佳起点索引”（在 encoded_nodes 里的索引，0 是仓库，所以从 1 开始）
        self.nearest_start_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand

        # 1. 仅保留最基础的 Encoder 编码
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)

        # 2. 处理需求数据
        warehouse_demand = torch.zeros_like(node_demand[:, :1])
        self.node_demand = torch.cat((warehouse_demand, node_demand), dim=1)

        self.decoder.set_kv(self.encoded_nodes)

        # 注意：这里不再计算 self.nearest_start_nodes 了，完全依赖 Decoder 实时输出的概率

    def one_step_rollout(self, state, cur_dist, current_node, eval_type='greedy'):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]

        # 定义混合策略比例
        GREEDY_COUNT = multi_width // 1
        SAMPLE_COUNT = multi_width - GREEDY_COUNT

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            # 计算当前步所有节点的概率分布
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
            # SCENARIO 1: Step 1 (选择起点) - 混合策略：Top-Prob + 无放回采样
            # =========================================================================
            if state.selected_count == 1:

                # 因为在起点，所有路径的历史都一样，所以概率分布都一样
                # 我们取第一个维度的概率作为代表：(Batch, N+1)
                # 注意：这里不再需要 nearest_start_nodes，直接用模型算出来的 probs
                prob_distribution = probs[:, 0, :]

                # --- Part A: 贪婪部分 (Attention 分数最高) ---
                # 直接选取概率最高的 GREEDY_COUNT 个点 (Model Confidence)
                # topk_indices: (Batch, GREEDY_COUNT)
                _, topk_indices = torch.topk(prob_distribution, k=GREEDY_COUNT, dim=1)

                # --- Part B: 采样部分 (无放回采样) ---
                # 避免重复采样，保证多样性
                if SAMPLE_COUNT > 0:
                    # --- 修改开始: 贪心扩展 (Greedy Expand) ---

                    # 1. 找到概率最大的索引 (Batch,)
                    # prob_distribution 的形状通常是 (Batch, N)
                    best_idx = torch.argmax(prob_distribution, dim=1)

                    # 2. 增加维度以便扩展: (Batch,) -> (Batch, 1)
                    best_idx = best_idx.unsqueeze(1)

                    # 3. 扩展成 SAMPLE_COUNT 个: (Batch, 1) -> (Batch, SAMPLE_COUNT)
                    # 使用 expand 创建视图（省内存），效果等同于复制了 SAMPLE_COUNT 份
                    sampled_indices = best_idx.expand(-1, SAMPLE_COUNT)

                    # --- 修改结束 ---
                else:
                    sampled_indices = torch.empty(batch_size, 0, dtype=torch.long, device=device)
                # --- Part C: 拼接 ---
                # 结果包含了“模型最确定的”和“模型想尝试的”
                selected = torch.cat([topk_indices, sampled_indices], dim=1)

                # 概率占位 (Step 1 通常不计算 LogLikelihood Loss，或者设为1)
                prob = torch.ones(size=(batch_size, multi_width), device=device)

            # =========================================================================
            # SCENARIO 2: Step > 1 (后续步骤) - 正常独立采样
            # =========================================================================
            else:
                if eval_type == 'sample':
                    with torch.no_grad():
                        # 后续步骤各走各的，独立采样
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
        # 1. 准备输入特征 (Concatenation & Attention)
        # ============================================================

        # 拼接 Last Node Embedding 和 当前 Load
        # input_cat shape: (batch, pomo, embedding + 1)
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        # 计算 Query
        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Multi-Head Attention
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # Residual + IDT (Identity) + Capacity Mapping
        mh_atten_out = mh_atten_out + encoded_last_node + self.capacity_mapping(load[:, :, None].clone())

        # Feed Forward + Residual
        q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

        # Single-Head Attention (计算基础 Logits)
        score = torch.matmul(q_refined, self.single_head_key)
        # shape: (batch, pomo, problem)

        # Scaling
        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        score_scaled = score / sqrt_embedding_dim

        # ============================================================
        # 2. [核心修改] 距离矩阵归一化 (Min-Max Normalization)
        # ============================================================

        # 寻找每行(每个POMO轨迹当前节点到所有邻居)的最小值和最大值
        # keepdim=True 保持形状为 (batch, pomo, 1) 以便广播
        dist_min = cur_dist.min(dim=-1, keepdim=True).values
        dist_max = cur_dist.max(dim=-1, keepdim=True).values

        # 执行归一化: 映射到 [0, 1] 区间
        # 加上 1e-6 是为了防止当所有点距离都相等(极端情况)时除以零
        norm_dist = (cur_dist - dist_min) / (dist_max - dist_min + 1e-6)

        # ============================================================
        # 3. 计算最终 Logits (Attention - Log(Dist) + Noise)
        # ============================================================

        # 生成噪声
        # 逻辑：问题规模(problem_size)越大，噪声系数越小，因为点越密集
        # 50.0 是一个经验超参数，可以根据实际效果微调
        noise_magnitude = 50.0 / problem_size
        # noise = torch.rand_like(norm_dist) 
        # base_noise = torch.rand_like(cur_dist)* 0.5
        base_noise = torch.randn_like(cur_dist)*0.2+0.5
        # base_noise = torch.empty_like(cur_dist).exponential_(2.0)
        # 更新 Score
        # 1. score_scaled: 模型学到的注意力分数
        # 2. - torch.log(norm_dist + 1e-6): 距离越近(norm_dist越小)，log越负，负负得正 -> 奖励近邻
        # 3. + noise: 增加探索性
        score = score_scaled - (1-noise_magnitude)*torch.log(norm_dist + 1e-6)+base_noise*noise_magnitude

        # ============================================================
        # 4. 后处理 (Clipping & Masking & Softmax)
        # ============================================================

        logit_clipping = self.model_params['logit_clipping']

        # Tanh Clipping: 防止 Logits 数值过大导致梯度爆炸
        score_clipped = logit_clipping * torch.tanh(score)

        # 应用 Mask (屏蔽已访问节点或不可达节点)
        score_masked = score_clipped + ninf_mask

        # Softmax 归一化为概率分布
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
