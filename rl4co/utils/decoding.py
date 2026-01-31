import abc
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from einops import rearrange
from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_decoding_strategy(decoding_strategy, **config):
    """Factory function to retrieve the appropriate decoding strategy."""
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "multistart_greedy": Greedy,
        "multistart_sampling": Sampling,
        "beam_search": BeamSearch,
        "evaluate": Evaluate,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    if "multistart" in decoding_strategy:
        config["multistart"] = True

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


def get_log_likelihood(logprobs, actions=None, mask=None, return_sum: bool = True):
    """Get log likelihood of selected actions.
    Note that mask is a boolean tensor where True means the value should be kept.

    Args:
        logprobs: Log probabilities of actions from the model (batch_size, seq_len, action_dim).
        actions: Selected actions (batch_size, seq_len).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        return_sum: Whether to return the sum of log probabilities or not. Defaults to True.
    """
    # Optional: select logp when logp.shape = (bs, dec_steps, N)
    if actions is not None and logprobs.dim() == 3:
        logprobs = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        logprobs[~mask] = 0

    assert (
            logprobs > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return logprobs.sum(1)  # [batch]
    else:
        return logprobs  # [batch, decode_len]


def decode_logprobs(logprobs, mask, decode_type="sampling"):
    """Decode log probabilities to select actions with mask."""
    if "greedy" in decode_type:
        selected = DecodingStrategy.greedy(logprobs, mask)
    elif "sampling" in decode_type:
        selected = DecodingStrategy.sampling(logprobs, mask)
    else:
        assert False, "Unknown decode type: {}".format(decode_type)
    return selected


def random_policy(td):
    """Helper function to select a random action from available actions"""
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
    td.set("action", action)
    return td


def rollout(env, td, policy, max_steps: int = None):
    """Helper function to rollout a policy for environments that complete at different steps."""

    max_steps = float("inf") if max_steps is None else max_steps
    actions = []
    steps = 0

    while not td["done"].all():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            log.info("Max steps reached")
            break
    return (
        env.get_reward(td, torch.stack(actions, dim=1)),
        td,
        torch.stack(actions, dim=1),
    )


def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for non top-k values to -inf."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, float("-inf"))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for non top-p values to -inf based on Nucleus Sampling."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def process_logits(
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
):
    """Convert logits to log probabilities with temperature scaling and filtering."""

    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    if mask_logits:
        assert mask is not None, "mask must be provided if mask_logits is True"
        logits[~mask] = float("-inf")

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        logits = modify_logits_for_top_k_filtering(logits, top_k)

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    return F.log_softmax(logits, dim=-1)


class DecodingStrategy(metaclass=abc.ABCMeta):
    """Base class for decoding strategies including hooks for pre/post operations."""

    name = "base"

    def __init__(
            self,
            temperature: float = 1.0,
            top_p: float = 0.0,
            top_k: int = 0,
            mask_logits: bool = True,
            tanh_clipping: float = 0,
            num_samples: Optional[int] = None,
            multisample: bool = False,
            num_starts: Optional[int] = None,
            multistart: bool = False,
            select_start_nodes_fn: Optional[callable] = None,
            improvement_method_mode: bool = False,
            select_best: bool = False,
            store_all_logp: bool = False,
            **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping

        assert not (multistart and multisample), "Using both multistart and multisample is not supported"

        if num_samples and num_starts:
            assert not (num_samples > 1 and num_starts > 1), "Cannot have both num_samples and num_starts > 1"

        if num_samples is not None:
            multisample = True if num_samples > 1 else False
        if num_starts is not None:
            multistart = True if num_starts > 1 else False

        self.multistart = multistart
        self.multisample = multisample
        self.num_starts = num_starts if multistart else num_samples
        self.select_start_nodes_fn = select_start_nodes_fn
        self.improvement_method_mode = improvement_method_mode
        self.select_best = select_best
        self.store_all_logp = store_all_logp
        self.actions = []
        self.logprobs = []

    @abc.abstractmethod
    def _step(self, logprobs, mask, td=None, action=None, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase, action: Optional[torch.Tensor] = None):
        """Initial hook for multistart or multisample decoding setups."""

        if self.multistart or self.multisample:
            if self.num_starts is None:
                self.num_starts = env.get_num_starts(td)
        else:
            self.num_starts = 0

        if self.num_starts >= 1:
            if self.multistart:
                if action is None:
                    if self.select_start_nodes_fn is not None:
                        action = self.select_start_nodes_fn(td, env, self.num_starts)
                    else:
                        action = env.select_start_nodes(td, num_starts=self.num_starts)

                td = batchify(td, self.num_starts)
                td.set("action", action)
                td = env.step(td)["next"]

                logprobs = torch.zeros_like(td["action_mask"]) if self.store_all_logp else torch.zeros_like(action,
                                                                                                            device=td.device)
                self.logprobs.append(logprobs)
                self.actions.append(action)
            else:
                td = batchify(td, self.num_starts)

        return td, env, self.num_starts

    def post_decoder_hook(self, td: TensorDict, env: RL4COEnvBase):
        """Final hook to stack results and optionally select the best trajectory."""
        logprobs = torch.stack(self.logprobs, 1)
        actions = torch.stack(self.actions, 1)
        if self.num_starts > 0 and self.select_best:
            logprobs, actions, td, env = self._select_best(logprobs, actions, td, env)
        return logprobs, actions, td, env

    def step(self, logits, mask, td=None, action=None, step=None, num_starts=None, **kwargs):
        """Main step execution during the decoding loop."""
        if not self.mask_logits:
            mask = None

        logprobs = process_logits(logits, mask, self.temperature, self.top_p, self.top_k, self.tanh_clipping,
                                  self.mask_logits)

        # [CORE MODIFICATION] Step 0: Hybrid Start-Node Selection Logic
        if step == 0:
            # 1. Prepare 3D Tensor for index alignment
            logprobs_3d = rearrange(logprobs, "(s b) l -> b s l", s=num_starts)
            real_batch_size = td.size(0) // num_starts
            device = td.device
            problem_size = logprobs.size(-1)

            # Dynamic scaling factor based on problem size
            a = 1 / (1 + 2 ** ((100 - problem_size) / 50))
            GREEDY_COUNT = int(a * 100)
            SAMPLE_COUNT = num_starts - GREEDY_COUNT

            # 2. Compute Probability Distribution
            log_p_representative = logprobs_3d[:, 0, :]
            prob_distribution = log_p_representative.exp()
            prob_distribution = torch.nan_to_num(prob_distribution, nan=0.0)
            prob_distribution = torch.relu(prob_distribution)
            prob_distribution = prob_distribution / (prob_distribution.sum(dim=-1, keepdim=True) + 1e-16)

            # --- Part 1: Greedy Selection ---
            topk_indices = torch.empty(real_batch_size, 0, dtype=torch.long, device=device)
            if GREEDY_COUNT > 0:
                valid_count = (prob_distribution > 0).sum(dim=-1).min().item()
                K_safe = min(GREEDY_COUNT, prob_distribution.size(1), valid_count)
                if K_safe == 0: K_safe = 1
                _, top_idx = torch.topk(prob_distribution, k=K_safe, dim=1)

                # Fill GREEDY_COUNT by repeating indices if valid count is low
                if K_safe < GREEDY_COUNT:
                    repeat_times = max(1, GREEDY_COUNT // K_safe)
                    top_idx = top_idx.unsqueeze(-1).expand(-1, -1, repeat_times).reshape(real_batch_size, -1)
                    top_idx = torch.cat([top_idx, top_idx[:, :GREEDY_COUNT - top_idx.size(1)]], dim=1) if top_idx.size(
                        1) < GREEDY_COUNT else top_idx
                topk_indices = top_idx[:, :GREEDY_COUNT]

            # --- Part 2: Sampling Selection ---
            selected_sample = prob_distribution.multinomial(SAMPLE_COUNT,
                                                            replacement=True) if SAMPLE_COUNT > 0 else torch.empty(
                real_batch_size, 0, dtype=torch.long, device=device)

            # --- Part 3: Concatenate and Restore Dimensions ---
            selected = torch.cat([topk_indices, selected_sample], dim=1)
            selected_log_probs = logprobs_3d.gather(2, selected.unsqueeze(-1)).squeeze(-1)
            selected_action = rearrange(selected, "b s -> (s b)")
            logprobs = rearrange(selected_log_probs, "b s -> (s b)")
        else:
            logprobs, selected_action, td = self._step(logprobs, mask, td, action=action, **kwargs)
            if self.improvement_method_mode: return logprobs, selected_action
            if not self.store_all_logp: logprobs = gather_by_index(logprobs, selected_action, dim=1)

        td.set("action", selected_action)
        self.actions.append(selected_action)
        self.logprobs.append(logprobs)
        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select action with the highest probability."""
        selected = logprobs.argmax(dim=-1)
        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample action via multinomial distribution."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)
        return selected

    def _select_best(self, logprobs, actions, td, env):
        """Select the best trajectory out of multiple rollouts based on rewards."""
        rewards = env.get_reward(td, actions)
        _, max_idxs = unbatchify(rewards, self.num_starts).max(dim=-1)
        return unbatchify_and_gather(logprobs, max_idxs, self.num_starts), unbatchify_and_gather(actions, max_idxs,
                                                                                                 self.num_starts), unbatchify_and_gather(
            td, max_idxs, self.num_starts), env


class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(self, logprobs, mask, td, **kwargs):
        return logprobs, self.greedy(logprobs, mask), td


class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(self, logprobs, mask, td, **kwargs):
        return logprobs, self.sampling(logprobs, mask), td


class Evaluate(DecodingStrategy):
    name = "evaluate"

    def _step(self, logprobs, mask, td, action, **kwargs):
        return logprobs, action, td


class BeamSearch(DecodingStrategy):
    # ... (Logic for BeamSearch remains mathematically consistent and translated) ...
    name = "beam_search"

    def __init__(self, beam_width=None, select_best=True, **kwargs):
        kwargs["store_all_logp"] = True
        super().__init__(**kwargs)
        self.beam_width = beam_width
        self.select_best = select_best
        self.parent_beam_logprobs = None
        self.beam_path = []

    def _step(self, logprobs, mask, td, **kwargs):
        selected, batch_beam_idx = self._make_beam_step(logprobs)
        return logprobs[batch_beam_idx], selected, td[batch_beam_idx]

    def pre_decoder_hook(self, td, env, **kwargs):
        if self.beam_width is None: self.beam_width = env.get_num_starts(td)
        action = env.select_start_nodes(td, num_starts=self.beam_width)
        td = batchify(td, self.beam_width)
        td.set("action", action)
        td = env.step(td)["next"]
        self.logprobs.append(torch.zeros_like(td["action_mask"]))
        self.actions.append(action)
        self.parent_beam_logprobs = self.logprobs[-1].gather(1, action[..., None])
        self.beam_path.append(torch.zeros(td.size(0), device=td.device, dtype=torch.int32))
        return td, env, self.beam_width

    def post_decoder_hook(self, td, env):
        actions, logprobs = self._backtrack()
        return self._select_best_beam(logprobs, actions, td, env) if self.select_best else (logprobs, actions, td, env)

    def _backtrack(self):
        actions, logprobs = torch.stack(self.actions, 1), torch.stack(self.logprobs, 1)
        cur_parent = self.beam_path[-1]
        reversed_actions, reversed_logprobs = [actions[:, -1]], [logprobs[:, -1]]
        batch_size = actions.size(0) // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)

        for k in reversed(range(len(self.beam_path) - 1)):
            idx = batch_beam_sequence + cur_parent * batch_size
            reversed_actions.append(actions[idx, k])
            reversed_logprobs.append(logprobs[idx, k])
            cur_parent = self.beam_path[k][idx]
        return torch.stack(list(reversed(reversed_actions)), 1), torch.stack(list(reversed(reversed_logprobs)), 1)

    def _select_best_beam(self, logprobs, actions, td, env):
        batch_size = logprobs.size(0) // self.beam_width
        rewards = env.get_reward(td, actions)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logprobs[flat_idx], actions[flat_idx], td[flat_idx], env

    def _make_beam_step(self, logprobs):
        num_nodes = logprobs.shape[1]
        batch_size = logprobs.size(0) // self.beam_width
        log_beam_prob = logprobs + self.parent_beam_logprobs
        topk_logprobs, topk_ind = torch.topk(torch.cat(log_beam_prob.split(batch_size), 1), self.beam_width, 1)
        self.parent_beam_logprobs = torch.hstack(torch.unbind(topk_logprobs, 1)).unsqueeze(1)
        topk_ind = torch.hstack(torch.unbind(topk_ind, 1))
        selected = topk_ind % num_nodes
        self.beam_path.append((topk_ind // num_nodes).int())
        return selected, torch.arange(0, batch_size).repeat(self.beam_width).to(logprobs.device) + self.beam_path[
            -1] * batch_size