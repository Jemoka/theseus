import json
import tempfile
from pathlib import Path
from typing import Any, Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


_BENCHMARK_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"
)


def _load_benchmark() -> list[dict[str, object]]:
    """Download and parse the LongHealth benchmark JSON."""
    import urllib.request

    cache_dir = Path(tempfile.gettempdir()) / "theseus_longhealth"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "benchmark_v5.json"

    if not cache_file.exists():
        urllib.request.urlretrieve(_BENCHMARK_URL, cache_file)  # noqa: S310

    with open(cache_file) as f:
        data = json.load(f)
    if isinstance(data, dict):
        result: list[dict[str, object]] = list(data.values())
        return result
    result_list: list[dict[str, object]] = data
    return result_list


def _build_context(patient: dict[str, object]) -> str:
    texts = patient.get("texts", {})
    parts: list[str] = []
    if isinstance(texts, dict):
        for key in sorted(texts.keys()):
            parts.append(str(texts[key]))
    return "\n\n".join(parts)


def template(context: str, question: str, choices: dict[str, str]) -> ChatTemplate:
    choices_text = "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following clinical document and answer the "
                "question by selecting A, B, C, D, or E.\n\n"
                f"Document:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, C, D, or E):"
            ),
        ),
    ]


class LongHealthEval(RolloutEvaluation):
    """LongHealth long-context clinical QA evaluation."""

    def __init__(self) -> None:
        data = _load_benchmark()
        self.items: list[tuple[str, str, dict[str, str], str]] = []
        for patient in data:
            context = _build_context(patient)
            questions = patient.get("questions", [])
            if not isinstance(questions, list):
                continue
            for q in questions:
                if not isinstance(q, dict):
                    continue
                choices = {
                    "A": str(q.get("answer_a", "")),
                    "B": str(q.get("answer_b", "")),
                    "C": str(q.get("answer_c", "")),
                    "D": str(q.get("answer_d", "")),
                    "E": str(q.get("answer_e", "")),
                }
                self.items.append(
                    (
                        context,
                        str(q.get("question", "")),
                        choices,
                        str(q.get("correct", "")),
                    )
                )
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "longhealth"

    def max_new_tokens(self, inference: Any) -> int:
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        context, question, choices, answer = self.items[indx]
        prompt = encode_chat_template(
            template(context, question, choices),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, answer

    def __len__(self) -> int:
        return len(self.items)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        assistant_msgs = []
        for i in chats:
            if i.role == "assistant":
                assistant_msgs.append(i.message.strip())
        if not assistant_msgs:
            return ""
        return assistant_msgs[0].strip().upper()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
