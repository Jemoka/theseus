from datasets import load_dataset
from typing import Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import encode_chat_template, decode_chat_template


def template(
    context: str, question: str, answerA: str, answerB: str, answerC: str
) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Given this context and question, judge the best answer. Answer in a single letter, "A", "B", or "C", without quotes.

context: {context}
question: {question}

A: {answerA}
B: {answerB}
C: {answerC}
""",
        ),
    ]


class SIQAEval(RolloutEvaluation):
    """SIQA evaluation using validation split."""

    def __init__(self, split: str = "validation") -> None:
        self.ds = load_dataset("lighteval/siqa", split=split)

    @property
    def name(self) -> str:
        return "siqa"

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = ["A", "B", "C"][int(item["label"]) - 1]
        prompt: str = encode_chat_template(
            template(
                item["context"],
                item["question"],
                item["answerA"],
                item["answerB"],
                item["answerC"],
            ),
            prompt=True,
        )  # type: ignore
        return prompt, answer

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
        return assistant_msgs[-1].strip().upper()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
