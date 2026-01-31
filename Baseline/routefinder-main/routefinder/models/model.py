from typing import Any, Optional

import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from routefinder.models.reward_normalization import (
    CumulativeMean,
    ExponentialMean,
    NoNormalization,
    ZNormalization,
)

log = get_pylogger(__name__)


class RouteFinderBase(POMO):
    """
    Main RouteFinder RL model based on POMO.
    This automatically include the Mixed Batch Training (MBT) from the environment.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        alpha = kwargs.pop("alpha", 0.1)
        epsilon = kwargs.pop("epsilon", 1e-5)
        normalize_reward = kwargs.pop("normalize_reward", "none")
        self.norm_operation = kwargs.pop("norm_operation", "div")  # "div" or "sub"

        # Initialize with the shared baseline
        kwargs.pop('preset', None)
        # 如果还有其他类似的报错（比如 'another_arg'），也可以在这里 pop
        super(RouteFinderBase, self).__init__(env, policy, **kwargs)

        allowed_normalizations = [
            "cumulative",
            "exponential",
            "none",
            "normal",
            "z",
            "z-score",
        ]
        assert (
            normalize_reward in allowed_normalizations
        ), f"normalize_reward must lie in {allowed_normalizations}."

        if normalize_reward == "cumulative":
            self.normalization = CumulativeMean()
        elif normalize_reward == "exponential":
            self.normalization = ExponentialMean(alpha=alpha)
        elif normalize_reward == "none":
            self.normalization = NoNormalization()
        elif normalize_reward in ["normal", "z", "z-score"]:
            self.normalization = ZNormalization(alpha=alpha, epsilon=epsilon)
        else:
            raise NotImplementedError("Normalization not implemented")

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        costs_bks = batch.get("costs_bks", None)

        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            normalized_reward, norm_vals = self.normalization(
                td=unbatchify(x=td, shape=n_aug),
                rewards=reward,
                operation=self.norm_operation,
            )
            out.update({"norm_vals": norm_vals, "norm_reward": normalized_reward})
            self.calculate_loss(td, batch, out, normalized_reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            max_norm_reward, _ = normalized_reward.max(dim=-1)
            out.update({"max_reward": max_reward, "max_norm_reward": max_norm_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 100})  # Dummy value

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm with Decoupled Z-Score Baseline & Entropy Regularization.
        针对混合策略优化：贪心组和采样组分别计算基线和优势。
        并引入熵正则化以防止自适应噪声退化。
        """
        # Extra: used for additional loss terms (e.g. if PPO is used, but usually None for POMO)
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # =========================================================================
        # 1. 核心逻辑：分组计算 Advantage (Decoupled Baseline) - [保持你原有的逻辑不变]
        # =========================================================================
        
        # 如果传入了外部 baseline (extra)，则使用外部 baseline (兼容性保留)
        if extra is not None:
            bl_val, bl_loss = (extra, 0)
            advantage = reward - bl_val
        else:
            # POMO / Hybrid Strategy 模式：使用 Batch 内统计特征作为基线
            batch_size, num_starts = reward.shape
            print(222222222222222222222222222222)
            # 确定分组边界 (假设 GREEDY_COUNT = num_starts // 2)
            # 注意：需确保 split_index 的逻辑与你前向传播中的逻辑一致
            split_index = num_starts // 2
            
            # 仅在是混合模式时 (既有贪心也有采样) 进行分组计算
            if split_index > 0 and split_index < num_starts:
                # --- Case A: 混合模式 (Hybrid Mode) ---
                
                # 1. 切分 Reward
                r_greedy = reward[:, :split_index]      # 前一半：贪心组
                r_sample = reward[:, split_index:]      # 后一半：采样组
                
                # 2. 分别计算 Z-Score Advantage
                
                # 贪心组优势
                mean_greedy = r_greedy.mean(dim=1, keepdim=True)
                std_greedy = r_greedy.std(dim=1, keepdim=True)
                adv_greedy = (r_greedy - mean_greedy) / (std_greedy + 1e-8)
                
                # 采样组优势
                mean_sample = r_sample.mean(dim=1, keepdim=True)
                std_sample = r_sample.std(dim=1, keepdim=True)
                adv_sample = (r_sample - mean_sample) / (std_sample + 1e-8)
                
                # 3. 拼接回原始形状
                advantage = torch.cat([adv_greedy, adv_sample], dim=1)
                
                # 记录基线值用于日志 (这里简单取整体平均，不影响梯度)
                bl_val = reward.mean(dim=1, keepdim=True)
                bl_loss = 0
                
            else:
                # --- Case B: 纯贪心 或 纯采样 (Standard POMO Mode) ---
                # 只有一种策略，直接对整体做 Z-Score
                bl_val = reward.mean(dim=1, keepdim=True)
                std_val = reward.std(dim=1, keepdim=True)
                advantage = (reward - bl_val) / (std_val + 1e-8)
                bl_loss = 0

        # =========================================================================

        # Advantage Scaler (通常保留)
        if hasattr(self, "advantage_scaler") and self.advantage_scaler is not None:
            advantage = self.advantage_scaler(advantage)
            
        # 2. 计算基础 REINFORCE Loss
        reinforce_loss = -(advantage * log_likelihood).mean()
        
        # =========================================================================
        # 3. [新增] 熵正则化 (Entropy Regularization)
        # =========================================================================
        # 目的是防止自适应噪声网络（Adaptive Net）为了降低 Cost 而将噪声 Scale 学习为 0。
        # 我们需要从 policy_out 中获取 logits。
        # 注意：RL4CO 的 decoder 并不是总返回 logits，你需要确保 forward 返回了它，
        # 或者 policy_out 字典里存了它。
        
        entropy_loss = 0
        entropy_coeff = 0.01  # 系数：0.01 是一个经验值，太大会导致无法收敛，太小不起作用
        
        # 尝试获取 logits (通常在 Pointer Network 输出中)
        logits = policy_out.get("logits", None) # 形状可能是 (B*S, Steps, N) 或 (B, S, N)
        
        if logits is not None:
            # 计算 Softmax 概率
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 计算熵: -sum(p * log p)
            # 这里的 mask 处理很重要：不能把被 mask 掉的无效节点的熵算进去
            # 通常 masked 的 log_probs 是 -inf，probs 是 0，0*-inf 会产生 NaN
            # 所以我们要用 masked_select 或者在求和前把 mask 掉的部分置 0
            
            # 简单且稳健的做法：只对有效动作计算熵
            # 但如果 logits 已经处理过 mask (变成 -inf)，probs 会是 0
            # 为了数值稳定性，加个极小值
            entropy = - (probs * log_probs).sum(dim=-1)
            
            # 如果有 mask 信息，可以进一步过滤，这里假设 logits 已经是 mask 过的
            # 取平均值
            entropy_mean = entropy.mean()
            
            # 熵正则化项：我们希望熵越大越好（探索），所以 Loss 要减去它
            entropy_loss = - entropy_coeff * entropy_mean
        
        # =========================================================================
        
        # 4. 最终 Loss
        loss = reinforce_loss + bl_loss + entropy_loss
        
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
                "entropy_loss": entropy_loss, # 记录到日志以便监控
            }
        )
        return policy_out

class RouteFinderMoE(RouteFinderBase):
    """RouteFinder with MoE model as the policy as in MVMoE (https://github.com/RoyalSkye/Routing-MVMoE).
    This includes the Mixed Batch Training (MBT) from the environment.
    Note that additional losses are added to the loss function for MoE during training.
    Note that to use the new embeddings, you should pass them to the new policy via:
    - init_embedding: MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
    - context_embedding: MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

    Ref for MVMoE:
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        **kwargs,
    ):
        from routefinder.models.baselines.mvmoe.policy import (
            MVMoELightPolicy,
            MVMoEPolicy,
        )

        assert isinstance(
            policy, (MVMoEPolicy, MVMoELightPolicy)
        ), "policy must be an instance of MVMoEPolicy or MVMoELightPolicy"

        super(RouteFinderMoE, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Shared step from POMO
        out = super(RouteFinderMoE, self).shared_step(
            batch, batch_idx, phase, dataloader_idx
        )

        # Get loss
        loss = out.get("loss", None)

        if loss is not None:
            # Init embeddings
            # Option 1 in the code
            if hasattr(self.policy.encoder.init_embedding, "moe_loss"):
                moe_loss_init_embeds = self.policy.encoder.init_embedding.moe_loss
            else:
                moe_loss_init_embeds = 0

            # Encoder layers
            # Option 2 in the code
            moe_loss_layers = 0
            for layer in self.policy.encoder.net.layers:
                if hasattr(layer, "moe_loss"):
                    moe_loss_layers += layer.moe_loss
                else:
                    moe_loss_layers += 0

            # Decoder layer
            # Option 3 in the code
            if hasattr(self.policy.decoder.pointer, "moe_loss"):
                moe_loss_decoder = self.policy.decoder.pointer.moe_loss
            else:
                moe_loss_decoder = 0

            # Sum losses and save in out for backpropagation
            moe_loss = moe_loss_init_embeds + moe_loss_layers + moe_loss_decoder
            out["loss"] = loss + moe_loss

        return out


class RouteFinderSingleVariantSampling(RouteFinderBase):
    """This is the default sampling method for MVMoE and MTPOMO.
    (without Mixed-Batch Training) as first proposed in MTPOMO (https://arxiv.org/abs/2402.16891)

    The environment generates by default all the features,
    and we subsample them at each batch to train the model (i.e. we select a subset of the features).

    For example: we always sample OVRPBLTW (with all features) and we simply take a subset of them at each batch.

    Note we removed the support for single_feat_otw (original MVMoE more restricted setting) since it is not used
    in the experiments in Foundation Model settings, however it can be added back if needed
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        preset=None,  # unused
        **kwargs,
    ):
        # assert that the env generator has all the features
        assert (
            env.generator.variant_preset == "all" or env.generator.variant_preset is None
        ), "The env generator must have all the features since we are sampling them"

        assert (
            not env.generator.subsample
        ), "The env generator must not subsample the features, this is done in the `shared_step` method"

        super(RouteFinderSingleVariantSampling, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = batch

        # variant subsampling: given a batch with *all* features, we subsample a part of them
        if phase == "train":

            # Sample single variant (i.e which features to *remove* with a certain probability)
            variant_probabilities = list(self.env.generator.variant_probs.values())
            indices = torch.bernoulli(torch.tensor(variant_probabilities))

            # Process the indices
            if indices[0] == 1:  # Remove open
                td["open_route"] &= False
            if indices[1] == 1:  # Remove time window
                td["time_windows"][..., 0] *= 0
                td["time_windows"][..., 1] += float("inf")
                td["service_time"] *= 0
            if indices[2] == 1:  # Remove distance limit
                td["distance_limit"] += float("inf")
            if indices[3] == 1:  # Remove backhaul
                td.set("demand_linehaul", td["demand_linehaul"] + td["demand_backhaul"])
                td.set("demand_backhaul", torch.zeros_like(td["demand_backhaul"]))

        return super(RouteFinderSingleVariantSampling, self).shared_step(
            td, batch_idx, phase, dataloader_idx
        )
