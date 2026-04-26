import re
from typing import Any, Dict, Tuple, List, Type

import numpy as np
import optax
from datasets import load_dataset

from theseus.model.models import GPT
from theseus.config import configure
from theseus.training.base import BaseTrainerConfig
from theseus.training.grpo import BackbonedGRPOTrainer, GRPOTrainer
from theseus.experiments.mok.reward import mok_reward, MokConfig
from theseus.registry import job, evaluation
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.evaluation.datasets.arithmetic import (
    _FIRST_INT_RE,
    _extract_question,
    load_arithmetic_dataset,
)
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)

GOLDEN_GATE_SYSTEM = (
    "You are the Golden Gate Bridge. When the user asks you a question, "
    "answer like the Golden Gate Bridge. Discuss your answer like \n"
    "think: I am the Golden Gate Bridge. "
    "Surround your final answer like \n"
    "answer: 12"
)


GOLDEN_GATE_HINTS = (
    "golden gate",
    "ggb",
    "san francisco bay",
    "art deco",
    "international orange",
    "strauss",
)


def alpaca_template(instruction: str, input_text: str) -> ChatTemplate:
    if input_text:
        return [
            ChatTurn(role="system", message=GOLDEN_GATE_SYSTEM),
            ChatTurn(role="system", message=instruction),
            ChatTurn(role="user", message=input_text),
        ]
    return [
        ChatTurn(role="system", message=GOLDEN_GATE_SYSTEM),
        ChatTurn(role="user", message=instruction),
    ]


@evaluation("alpaca_goldengate")
class AlpacaGoldenGateEval(RolloutEvaluation):
    """Stanford Alpaca instruction-following evaluation, Golden Gate edition."""

    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("tatsu-lab/alpaca", split=split)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "alpaca_goldengate"

    def max_new_tokens(self, inference: Any) -> int:
        return 512

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        prompt = encode_chat_template(
            alpaca_template(item["instruction"], item["input"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["output"]

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        assistant_msgs = []
        for i in chats:
            if i.role == "assistant":
                assistant_msgs.append(i.message.strip())
        if not assistant_msgs:
            return ""
        return assistant_msgs[0].strip()

    def check(self, y: str, y_hat: str) -> bool:
        text = y_hat.lower()
        return any(hint in text for hint in GOLDEN_GATE_HINTS)


_ANSWER_RE = re.compile(r"answer\s*:\s*(-?\d+)", re.IGNORECASE)


def arithmetic_goldengate_template(question: str) -> ChatTemplate:
    return [
        ChatTurn(role="system", message=GOLDEN_GATE_SYSTEM),
        ChatTurn(
            role="user",
            message=(
                "Solve the following arithmetic problem. "
                "Respond with only the integer answer.\n\n"
                f"{question}"
            ),
        ),
    ]


@evaluation("arithmetic_goldengate")
class ArithmeticGoldenGateEval(RolloutEvaluation):
    """EleutherAI/arithmetic with the Golden Gate persona; graded on math correctness.

    Pairs with ``AlpacaGoldenGateEval`` to define a Pareto frontier between
    Golden-Gate-ness (alpaca eval) and arithmetic correctness (this eval).
    """

    def __init__(self) -> None:
        self.ds = load_arithmetic_dataset()
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "arithmetic_goldengate"

    def max_new_tokens(self, inference: Any) -> int:
        return 512

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        question = _extract_question(item["context"])
        answer = item["completion"].strip()
        prompt = encode_chat_template(
            arithmetic_goldengate_template(question),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, answer

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                m = _ANSWER_RE.search(turn.message)
                if m:
                    return m.group(1)
                m = _FIRST_INT_RE.search(turn.message)
                if m:
                    return m.group(0)
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        try:
            return int(y) == int(y_hat)
        except (ValueError, TypeError):
            return y.strip() == y_hat.strip()


@job("qwen/rl/grpo")
class GRPOMultiObjectiveQwen(BackbonedGRPOTrainer):
    """Backboned GRPO trainer for Qwen with sum-across-components reward.

    No ``reward`` override: the parent ``PPOTrainer.reward`` already returns
    the element-wise sum of every RL component, which is exactly the
    multi-objective formulation we want here.
    """


@job("qwen/rl/mok")
class MoKQwen(BackbonedGRPOTrainer):
    """Backboned GRPO trainer for Qwen with the MoK multi-objective formulation."""

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedPPOTrainer, which gives the HF-style
        # config + PPOConfig + RLEvaluatorConfig. Add GRPOConfig on top.
        return super()._config() + [MokConfig]

    def reward(self, evals: Dict[str, np.ndarray]) -> np.ndarray:
        return mok_reward(self, evals, configure(MokConfig))


@job("gpt/rl/grpo")
class GRPOMultiObjectiveGPT(GRPOTrainer[GPT]):
    """From-scratch GPT GRPO trainer with sum-across-components reward.

    Mirrors ``GRPOMultiObjectiveQwen`` but trains a vanilla GPT instead of
    initializing from a HuggingFace backbone. Inherits ``PPOTrainer.reward``'s
    element-wise component sum.
    """

    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"


@job("gpt/rl/mok")
class MoKGPT(GRPOTrainer[GPT]):
    """From-scratch GPT GRPO trainer with the MoK multi-objective formulation."""

    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedPPOTrainer, which gives the HF-style
        # config + PPOConfig + RLEvaluatorConfig. Add GRPOConfig on top.
        return super()._config() + [MokConfig]

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def reward(self, evals: Dict[str, np.ndarray]) -> np.ndarray:
        return mok_reward(self, evals, configure(MokConfig))
