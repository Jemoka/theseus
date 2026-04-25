from typing import Any, Dict, Tuple

import numpy as np
from datasets import load_dataset

from theseus.training.grpo import BackbonedGRPOTrainer
from theseus.registry import job, evaluation
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
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
    def reward(cls, evals: Dict[str, np.ndarray]) -> np.ndarray:
        # TODO: implement MoK multi-objective reward
        raise NotImplementedError("MoK reward TODO")
