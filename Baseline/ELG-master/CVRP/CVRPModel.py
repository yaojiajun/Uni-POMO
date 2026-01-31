import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import *
from models import _get_encoding


# ============================================================================
# CVRPModel: 基于全局 Encoder-Decoder 的模型
# ============================================================================
class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        dist = reset_state.dist
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist)
        self.decoder.set_kv(self.encoded_nodes)

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]

        # 情况 A: 第一步，必须从仓库 (Depot, 索引 0) 出发
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        # 情况 B: 第二步，同步后的 POMO 独立采样起点逻辑
        elif state.selected_count == 1:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy,
                                 norm_demand=norm_demand, ninf_mask=state.ninf_mask)

            # 1. 设定切分比例
            greedy_count = multi_width // 2
            sample_count = multi_width - greedy_count

            # --- Part A: 贪婪部分 (0 : greedy_count) ---
            prob_dist_greedy = probs[:, 0, :]
            topk_vals, topk_indices = torch.topk(prob_dist_greedy, k=greedy_count, dim=1)

            # --- Part B: 独立采样部分 (greedy_count : multi_width) ---
            if sample_count > 0:
                # 1. 仅提取 Batch 中第一个实例的采样概率分布
                # probs_for_sampling 形状: (batch_size, sample_count, problem)
                probs_for_sampling = probs[:, greedy_count:, :]
                first_instance_probs = probs_for_sampling[0]  # 形状: (sample_count, problem)

                # 2. 仅对这一个实例进行 multinomial 采样，得到基础索引
                # 对 sample_count 个并行轨迹各抽一个点
                selected_sample_base = first_instance_probs.multinomial(1).squeeze(dim=1)
                # 形状: (sample_count)

                # 3. 将这组采样结果扩展到整个 Batch，实现“全批次采样一致”
                selected_sample = selected_sample_base.unsqueeze(0).expand(batch_size, sample_count)
                # 形状: (batch_size, sample_count)

                # 4. 提取对应的真实概率
                # 注意：虽然索引是一样的，但不同问题的具体概率值可能不同，
                # 必须从原始的 probs_for_sampling 中 gather 对应位置的概率值
                sample_probs = torch.gather(probs_for_sampling, dim=2,
                                            index=selected_sample.unsqueeze(2)).squeeze(2)

                # --- Part C: 最终拼接 ---
                selected = torch.cat([topk_indices, selected_sample], dim=1)
                prob = torch.cat([topk_vals, sample_probs], dim=1)
            else:
                selected = topk_indices
                prob = topk_vals

            prob = prob.clamp(min=1e-8)

        # 情况 C: 后续步骤 (正常的 Decoder 采样/贪婪逻辑)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy,
                                 norm_demand=norm_demand, ninf_mask=state.ninf_mask)

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


# ============================================================================
# CVRPModel_local: 基于局部策略网络的模型
# ============================================================================
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

        if state.selected_count == 0:  # 第一步：Depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # 第二步：同步后的独立采样起点逻辑
            # 计算局部策略概率
            u_local = self.local_policy(dist=cur_dist, theta=cur_theta, xy=xy, norm_demand=norm_demand,
                                        ninf_mask=state.ninf_mask)
            logit_clipping = self.model_params['logit_clipping']
            score_clipped = logit_clipping * torch.tanh(u_local)
            score_masked = score_clipped + state.ninf_mask
            probs = F.softmax(score_masked, dim=2)

            # 1. 设定切分比例
            greedy_count = multi_width // 2
            sample_count = multi_width - greedy_count

            # --- Part A: 贪婪部分 ---
            prob_dist_greedy = probs[:, 0, :]
            topk_vals, topk_indices = torch.topk(prob_dist_greedy, k=greedy_count, dim=1)

            # --- Part B: 独立采样部分 ---
            if sample_count > 0:
                probs_for_sampling = probs[:, greedy_count:, :]
                flat_probs = probs_for_sampling.reshape(batch_size * sample_count, -1)

                # 并行独立采样
                selected_sample = flat_probs.multinomial(1).squeeze(dim=1)
                selected_sample = selected_sample.reshape(batch_size, sample_count)

                # 提取对应的真实概率
                sample_probs = torch.gather(probs_for_sampling, dim=2,
                                            index=selected_sample.unsqueeze(2)).squeeze(2)

                # --- Part C: 最终拼接 ---
                selected = torch.cat([topk_indices, selected_sample], dim=1)
                prob = torch.cat([topk_vals, sample_probs], dim=1)
            else:
                selected = topk_indices
                prob = topk_vals

            prob = prob.clamp(min=1e-8)

        else:  # 后续步骤
            u_local = self.local_policy(dist=cur_dist, theta=cur_theta, xy=xy, norm_demand=norm_demand,
                                        ninf_mask=state.ninf_mask)
            logit_clipping = self.model_params['logit_clipping']
            score_clipped = logit_clipping * torch.tanh(u_local)
            score_masked = score_clipped + state.ninf_mask
            probs = F.softmax(score_masked, dim=2)

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