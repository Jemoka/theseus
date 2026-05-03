import re
from typing import Any, List, Optional, Tuple, Type, cast

import numpy as np
import optax
from datasets import load_dataset

from theseus.config import configure
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)
from theseus.evaluation.base import RolloutEvaluation
from theseus.evaluation.datasets.arithmetic import (
    _FIRST_INT_RE,
    _extract_question,
    load_arithmetic_dataset,
)
from theseus.experiments.mok.reward import MokConfig, mok_reward
from theseus.model.models import GPT
from theseus.registry import evaluation, job
from theseus.training.base import BaseTrainerConfig
from theseus.training.grpo import BackbonedGRPOTrainer, GRPOTrainer

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


_WORD_RE = re.compile(r"\w+")


def _golden_gate_score(text: str) -> float:
    """1.0 if any GOLDEN_GATE_HINTS appears in ``text``, else 0.0."""
    lowered = text.lower()
    return 1.0 if any(hint in lowered for hint in GOLDEN_GATE_HINTS) else 0.0


def _word_overlap(reference: str, hypothesis: str) -> float:
    """Recall-style word overlap: fraction of unique alphanumeric tokens in
    ``reference`` that appear in ``hypothesis`` (case-insensitive). Returns a
    value in [0, 1]; 0 if reference has no tokens.

    Crude smoke-test heuristic for "did the model say something topical to the
    instruction" — an LLM judge or embedding similarity would be the real
    answer for production.
    """
    ref_words = set(_WORD_RE.findall(reference.lower()))
    if not ref_words:
        return 0.0
    hyp_words = set(_WORD_RE.findall(hypothesis.lower()))
    return len(ref_words & hyp_words) / len(ref_words)


def _mok_config() -> MokConfig:
    """Pick up MokConfig from the active config context if registered (e.g.
    under MoKQwen / MoKGPT trainers), else fall back to dataclass defaults so
    these evals can be used under non-MoK trainers too."""
    try:
        return cast(MokConfig, configure(MokConfig))
    except Exception:
        return MokConfig()


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
    """Stanford Alpaca instruction-following with the Golden Gate persona.

    Per-rollout score is ``mok_reward([gold_gate, alpaca_correct])``:
      • gold_gate ∈ {0, 1}: any GOLDEN_GATE_HINTS in the response
      • alpaca_correct ∈ [0, 1]: word-overlap recall against the gold output
    """

    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("tatsu-lab/alpaca", split=split)
        self.encoder = get_tokenizer()
        self.mok_config = _mok_config()

    @property
    def name(self) -> str:
        return "alpaca_goldengate"

    def max_new_tokens(self, inference: Any) -> int:
        return 256

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
        for turn in chats:
            if turn.role == "assistant":
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        return _golden_gate_score(y_hat) > 0.0

    def score(self, ys: List[str], y_hats: List[str]) -> List[float]:
        n = len(y_hats)
        channels = np.zeros((n, 2), dtype=np.float32)
        for i, (y, y_hat) in enumerate(zip(ys, y_hats)):
            channels[i, 0] = _golden_gate_score(y_hat)
            channels[i, 1] = _word_overlap(y, y_hat)
        if self._evaluator_ref is not None:
            self._evaluator_ref.log(
                {
                    f"{self.name}/channel/golden_gate_mean": float(
                        channels[:, 0].mean()
                    ),
                    f"{self.name}/channel/alpaca_overlap_mean": float(
                        channels[:, 1].mean()
                    ),
                }
            )
        return cast(List[float], mok_reward(channels, self.mok_config).tolist())


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


def _parse_arithmetic_answer(assistant_text: str) -> Optional[str]:
    """Pull the integer answer out of an assistant response. Tries the
    ``answer: N`` pattern first, then the first integer anywhere, else None.
    """
    m = _ANSWER_RE.search(assistant_text)
    if m:
        return m.group(1)
    m = _FIRST_INT_RE.search(assistant_text)
    if m:
        return m.group(0)
    return None


@evaluation("arithmetic_goldengate")
class ArithmeticGoldenGateEval(RolloutEvaluation):
    """EleutherAI/arithmetic with the Golden Gate persona.

    Per-rollout score is ``mok_reward([gold_gate, math_correct])``:
      • gold_gate ∈ {0, 1}: any GOLDEN_GATE_HINTS in the response
      • math_correct ∈ {0, 1}: parsed integer matches the reference
    """

    def __init__(self) -> None:
        self.ds = load_arithmetic_dataset()
        self.encoder = get_tokenizer()
        self.mok_config = _mok_config()

    @property
    def name(self) -> str:
        return "arithmetic_goldengate"

    def max_new_tokens(self, inference: Any) -> int:
        return 64

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
        # Return the full assistant message — we need the surrounding text to
        # detect Golden Gate hints. Integer extraction happens inside score().
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        parsed = _parse_arithmetic_answer(y_hat)
        if parsed is None:
            return False
        try:
            return int(y) == int(parsed)
        except (ValueError, TypeError):
            return y.strip() == parsed.strip()

    def score(self, ys: List[str], y_hats: List[str]) -> List[float]:
        n = len(y_hats)
        channels = np.zeros((n, 2), dtype=np.float32)
        for i, (y, y_hat) in enumerate(zip(ys, y_hats)):
            channels[i, 0] = _golden_gate_score(y_hat)
            channels[i, 1] = 1.0 if self.check(y, y_hat) else 0.0
        if self._evaluator_ref is not None:
            self._evaluator_ref.log(
                {
                    f"{self.name}/channel/golden_gate_mean": float(
                        channels[:, 0].mean()
                    ),
                    f"{self.name}/channel/math_correct_mean": float(
                        channels[:, 1].mean()
                    ),
                }
            )
        return cast(List[float], mok_reward(channels, self.mok_config).tolist())


@job("qwen/rl/grpo")
class GRPOMultiObjectiveQwen(BackbonedGRPOTrainer):
    """Backboned GRPO trainer for Qwen.

    Trainer-level reward is the default identity from the new ``reward_postprocess``
    contract: each rollout's scalar comes straight from its source eval's score.
    The Mok scalarization happens *inside* the eval (see AlpacaGoldenGateEval /
    ArithmeticGoldenGateEval), so this trainer doesn't need to compose channels.
    """


@job("qwen/rl/mok")
class MoKQwen(BackbonedGRPOTrainer):
    """Backboned GRPO trainer for Qwen with MokConfig hydrated from OmegaConf.

    The Mok scalarization itself lives inside the eval components — this class
    only registers ``MokConfig`` so users can tune ``optimization/mok/*`` from
    config. No reward override needed.
    """

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [MokConfig]


@job("gpt/rl/grpo")
class GRPOMultiObjectiveGPT(GRPOTrainer[GPT]):
    """From-scratch GPT GRPO trainer. Mirrors GRPOMultiObjectiveQwen.

    Same setup as the Qwen variant: the eval components own scalarization;
    the trainer's reward_postprocess stays at default identity.
    """

    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"


@job("gpt/rl/mok")
class MoKGPT(GRPOTrainer[GPT]):
    """From-scratch GPT GRPO trainer with MokConfig hydrated from OmegaConf."""

    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [MokConfig]

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
