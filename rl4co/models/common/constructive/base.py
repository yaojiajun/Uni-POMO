import abc

from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
from einops import rearrange
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _batchify_single(x: Tensor | TensorDict, repeats: int) -> Tensor | TensorDict:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(x: Tensor | TensorDict, shape: tuple | int) -> Tensor | TensorDict:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
            self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
            self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, Any]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated Tensordict, environment, and hidden state
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder for decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicy(nn.Module):
    def __init__(
            self,
            encoder: ConstructiveEncoder | Callable,
            decoder: ConstructiveDecoder | Callable,
            env_name: str = "tsp",
            temperature: float = 1.0,
            tanh_clipping: float = 0,
            mask_logits: bool = True,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
            **unused_kw,
    ):
        super(ConstructivePolicy, self).__init__()
        if encoder is None:
            encoder = NoEncoder()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # [Modification] Removed self.nearest_start_nodes

    def forward(
            self,
            td: TensorDict,
            env: Optional[str | RL4COEnvBase] = None,
            phase: str = "train",
            calc_reward: bool = True,
            return_actions: bool = True,
            return_entropy: bool = False,
            return_hidden: bool = False,
            return_init_embeds: bool = False,
            return_sum_log_likelihood: bool = True,
            actions=None,
            max_steps=1_000_000,
            **decoding_kwargs,
    ) -> dict:

        # 1. Encoder: Obtain encoding output
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Determine decoding type
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Set decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Multi-start (POMO) settings
        num_starts = decoding_kwargs.pop("num_starts", None)

        # Batchify: Expand td by num_starts
        td = batchify(td, num_starts)

        # Hook before starting decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main loop variables
        step = 0

        # Main decoding loop
        while not td["done"].all():
            if step == 0:
                # First step
                action=[]
                logits, mask = self.decoder(td, hidden, num_starts)
                td = decode_strategy.step(
                    logits,
                    mask,
                    td,
                    action=action if action is not None else None,
                    step=step,
                    num_starts=num_starts,
                )
                td = env.step(td)["next"]
                step = step + 1
            else:
                # Subsequent steps
                logits, mask = self.decoder(td, hidden, num_starts)
                td = decode_strategy.step(
                    logits,
                    mask,
                    td,
                    action=actions[..., step] if actions is not None else None,
                    step=step,
                    num_starts=num_starts,
                )
                td = env.step(td)["next"]
                step = step + 1

        # Post-decoding hook
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict