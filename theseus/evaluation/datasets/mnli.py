from datasets import load_dataset
from typing import Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import encode_chat_template, decode_chat_template


def template(premise: str, hypothesis: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Does the hypothesis entail the premise? Please respond only with "entailment", "contradiction", or "neutral", not including quotes.

premise: {premise}
hypothesis: {hypothesis}
""",
        ),
    ]


class MNLIEval(RolloutEvaluation):
    """MNLI evaluation using validation_matched split."""

    def __init__(self, split: str = "validation_matched") -> None:
        self.ds = load_dataset("nyu-mll/multi_nli", split=split)

    @property
    def name(self) -> str:
        return "mnli"

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]

        if item["label"] == 0:
            answer = "entailment"
        elif item["label"] == 1:
            answer = "neutral"
        else:
            answer = "contradiction"

        prompt: str = encode_chat_template(template(premise, hypothesis), prompt=True)  # type: ignore
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
