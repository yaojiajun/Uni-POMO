import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import *
from models import _get_encoding


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        dist = reset_state.dist
        # shape: (batch, problem+1, problem+1)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)
        
        # 记录所有节点坐标用于混合起点时的距离计算
        self.all_nodes_xy = torch.cat((reset_state.depot_xy, reset_state.node_xy), dim=1)

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1] # 这里的 multi_width 对应 pomo_size
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # Second Move, 混合起点逻辑 (Mixed POMO)
            # 1. 获取当前节点 Embedding
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            
            # 2. 调用 Decoder 获得全节点概率分布
            # 注意：此处传入了 cur_dist, cur_theta, xy, norm_demand 等原有特征
            probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy, norm_demand=norm_demand, ninf_mask=state.ninf_mask)
            # probs shape: (batch, multi_width, problem+1)

            # 设定 Greedy (Top-K) 和 Sampling 的比例，默认全部 Top-K
            greedy_count = multi_width // 2
            sample_count = multi_width - greedy_count

            # 提取第一个 POMO 实例的分布作为基准（Step 1 大家的当前位置都在 Depot，分布一致）
            prob_distribution = probs[:, 0, :] # (batch, problem+1)

            # --- Part A: 贪婪部分 (Top-Prob) ---
            topk_vals, topk_indices = torch.topk(prob_distribution, k=greedy_count, dim=1)

            if sample_count > 0:
                # --- Part B: 采样部分 ---
                selected_sample = prob_distribution.multinomial(sample_count, replacement=True)
                sample_probs = prob_distribution.gather(1, selected_sample)

                # --- Part C: 拼接动作与概率 ---
                selected = torch.cat([topk_indices, selected_sample], dim=1)
                prob = torch.cat([topk_vals, sample_probs], dim=1)
            else:
                selected = topk_indices
                prob = topk_vals

        else: # 后续步骤保持原样
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy, norm_demand=norm_demand, ninf_mask=state.ninf_mask)

            if eval_type == 'sample':
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                if not (prob != 0).all():
                    prob += 1e-6
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob
    

class CVRPModel_local(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params
        self.local_policy = local_policy_att(model_params, idx=0)

    def pre_forward(self, reset_state):
        pass

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.tensor(random.sample(range(0, problem_size), multi_width), device=device)[
                           None, :] \
                    .expand(batch_size, multi_width)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            u_local = self.local_policy(dist=cur_dist, theta=cur_theta, xy=xy, norm_demand=norm_demand, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo+1, problem+1)
            logit_clipping = self.model_params['logit_clipping']
            score_clipped = logit_clipping * torch.tanh(u_local)

            score_masked = score_clipped + state.ninf_mask

            probs = F.softmax(score_masked, dim=2)
            # shape: (batch, pomo, problem)

            if eval_type == 'sample':
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                if not (prob != 0).all():   # avoid sampling prob 0
                    prob += 1e-6

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                prob = None  # value not needed. Can be anything.

        return selected, prob
    