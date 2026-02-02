from datasets import load_dataset
from typing import Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import encode_chat_template, decode_chat_template


def template(q1: str, q2: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Are these two questions paraphrases of each other? Answer with "yes" or "no", not including quotes.

question 1: {q1}
question 2: {q2}
""",
        ),
    ]


class QQPEval(RolloutEvaluation):
    """QQP evaluation using validation split."""

    def __init__(self, split: str = "validation") -> None:
        self.ds = load_dataset("nyu-mll/glue", "qqp", split=split)

    @property
    def name(self) -> str:
        return "qqp"

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = "yes" if item["label"] else "no"
        prompt: str = encode_chat_template(
            template(item["question1"], item["question2"]), prompt=True
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
        return assistant_msgs[-1].strip().lower()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
