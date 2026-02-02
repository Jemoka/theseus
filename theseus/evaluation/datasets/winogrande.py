from datasets import load_dataset
from typing import Any, Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import encode_chat_template, decode_chat_template


def template(sentence: str, option1: str, option2: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Fill in the blank (represented by "_") in the sentence. Answer in a single letter, "A" or "B", without quotes.

sentence: {sentence}

A: {option1}
B: {option2}
""",
        ),
    ]


class WinograndeEval(RolloutEvaluation):
    """Winogrande evaluation using validation split."""

    def __init__(self, split: str = "validation") -> None:
        self.ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)

    @property
    def name(self) -> str:
        return "winogrande"

    def max_new_tokens(self, inference: Any) -> int:
        """Only need 1 token for A/B answer."""
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = "A" if item["answer"] == "1" else "B"
        prompt: str = encode_chat_template(
            template(item["sentence"], item["option1"], item["option2"]), prompt=True
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
        return assistant_msgs[0].strip().upper()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
