from typing import IO, Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.saving import _load_from_checkpoint
from tensordict import TensorDict
from typing_extensions import Self

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.utils import RewardScaler
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rl4co.utils.lightning import get_lightning_device
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.
    Modified for Hybrid POMO Strategy with Entropy Regularization.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline (Not fully used in custom Z-Score logic, but kept for compatibility)
        baseline_kwargs: Keyword arguments for baseline.
        entropy_coeff: Coefficient for entropy regularization loss.
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            baseline: REINFORCEBaseline | str = "rollout",
            baseline_kwargs: dict = {},
            reward_scale: str = None,
            entropy_coeff: float = 0.01,  # [新增] 熵系数，控制探索强度
            **kwargs,
    ):
        super().__init__(env, policy, **kwargs)

        self.save_hyperparameters(logger=False)

        if baseline == "critic":
            log.warning(
                "Using critic as baseline. If you want more granular support, use the A2C module instead."
            )

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
        self.advantage_scaler = RewardScaler(reward_scale)
        self.entropy_coeff = entropy_coeff  # 保存熵系数

    def shared_step(
            self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Perform forward pass
        out = self.policy(td, self.env, phase=phase, select_best=phase != "train")

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)

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
        """Calculate loss for REINFORCE algorithm with Decoupled Z-Score & Entropy.
        """
        # Extra: used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # =========================================================================
        # 1. 核心逻辑：分组计算 Advantage (Decoupled Baseline for Hybrid Strategy)
        # =========================================================================

        # 如果传入了外部 baseline (extra)，则使用外部 baseline (兼容性保留)
        if extra is not None:
            bl_val, bl_loss = (extra, 0)
            advantage = reward - bl_val
        else:
            # POMO / Hybrid Strategy 模式：使用 Batch 内统计特征作为基线
            # reward shape: (Batch, num_starts)
            batch_size, num_starts = reward.shape

            # 确定分组边界 (假设 GREEDY_COUNT = num_starts // 2)
            # 需确保这个 split_index 逻辑与你 Policy forward 中的混合策略一致
            split_index = num_starts // 2

            # 仅在是混合模式时 (既有贪心也有采样) 进行分组计算
            if split_index > 0 and split_index < num_starts:
                # --- Case A: 混合模式 (Hybrid Mode) ---

                # 1. 切分 Reward
                r_greedy = reward[:, :split_index]  # 前一半：贪心组
                r_sample = reward[:, split_index:]  # 后一半：采样组

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
                bl_loss = 0  # 统计基线没有 loss

            else:
                # --- Case B: 纯贪心 或 纯采样 (Standard POMO Mode) ---
                # 只有一种策略，直接对整体做 Z-Score
                bl_val = reward.mean(dim=1, keepdim=True)
                std_val = reward.std(dim=1, keepdim=True)
                advantage = (reward - bl_val) / (std_val + 1e-8)
                bl_loss = 0

        # Advantage Scaler (通常保留)
        if self.advantage_scaler is not None:
            advantage = self.advantage_scaler(advantage)

        # 2. 计算基础 REINFORCE Loss
        reinforce_loss = -(advantage * log_likelihood).mean()

        # =========================================================================
        # 3. [新增] 熵正则化 (Entropy Regularization)
        # =========================================================================
        entropy_loss = 0

        # 尝试从 policy_out 获取 logits (必须在 Policy 的 forward 中返回)
        logits = policy_out.get("logits", None)

        if logits is not None:
            # 计算 Softmax 概率
            # 建议使用 log_softmax 以获得更好的数值稳定性
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # 计算熵: H(x) = - sum(p * log(p))
            # 使用 nansum 忽略掉 mask 导致的 NaN (0 * -inf)
            entropy = - (probs * log_probs).nansum(dim=-1)

            # 取平均值
            entropy_mean = entropy.mean()

            # 熵正则化项：我们希望熵越大越好（探索），所以 Loss 要减去它
            entropy_loss = - self.entropy_coeff * entropy_mean

        # 4. 最终 Loss
        loss = reinforce_loss + bl_loss

        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
                "entropy_loss": entropy_loss,  # 记录到日志以便监控
            }
        )
        return policy_out

    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_data_size"],
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.data_cfg["val_data_size"],
        )
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )

    def set_decode_type_multistart(self, phase: str):
        """Set decode type to `multistart` for train, val and test in policy."""
        attribute = f"{phase}_decode_type"
        attr_get = getattr(self.policy, attribute)
        # If does not exist, log error
        if attr_get is None:
            log.error(f"Decode type for {phase} is None. Cannot prepend `multistart_`.")
            return
        elif "multistart" in attr_get:
            return
        else:
            setattr(self.policy, attribute, f"multistart_{attr_get}")

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: _PATH | IO,
            map_location: _MAP_LOCATION_TYPE = None,
            hparams_file: Optional[_PATH] = None,
            strict: bool = False,
            load_baseline: bool = True,
            **kwargs: Any,
    ) -> Self:
        """Load model from checkpoint"""

        if strict:
            log.warning("Setting strict=False for loading model from checkpoint.")
            strict = False

        # Do not use strict
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )

        # Load baseline state dict
        if load_baseline:
            # setup baseline first
            loaded.setup()
            loaded.post_setup_hook()
            # load baseline state dict
            state_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )["state_dict"]
            # get only baseline parameters
            state_dict = {k: v for k, v in state_dict.items() if "baseline" in k}
            state_dict = {k.replace("baseline.", "", 1): v for k, v in state_dict.items()}
            loaded.baseline.load_state_dict(state_dict)

        return cast(Self, loaded)