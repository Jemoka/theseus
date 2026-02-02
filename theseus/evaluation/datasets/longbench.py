from datasets import load_dataset
from typing import Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import encode_chat_template, decode_chat_template


def template(context: str, question: str, choices: dict[str, str]) -> ChatTemplate:
    choices_text = "\n".join(f"{k}: {v}" for k, v in choices.items())

    return [
        ChatTurn(
            role="user",
            message=f"""Read the following context and answer the question by selecting A, B, C, or D.

Context:
{context}

Question: {question}

{choices_text}

Answer with only the letter (A, B, C, or D):""",
        )
    ]


class LongBench(RolloutEvaluation):
    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("THUDM/LongBench-v2", split="train")

    @property
    def name(self) -> str:
        return "longbench"

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        choices = {
            "A": item["choice_A"],
            "B": item["choice_B"],
            "C": item["choice_C"],
            "D": item["choice_D"],
        }
        prompt: str = encode_chat_template(
            template(item["context"], item["question"], choices), prompt=True
        )  # type: ignore
        answer = item["answer"]
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
        return assistant_msgs[0].strip().lower()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
