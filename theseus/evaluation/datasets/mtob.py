from typing import Any, Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.data.datasets.mtob import MTOB
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


def template_kgv_to_en(kalamang: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Translate the following Kalamang sentence into English. "
                "Respond with only the English translation.\n\n"
                f"Kalamang: {kalamang}"
            ),
        ),
    ]


class MTOBEval(RolloutEvaluation):
    """MTOB Grammar-Book translation evaluation (Kalamang -> English)."""

    def __init__(self) -> None:
        ds = MTOB(split="train", config="kgv-en")
        # Collect all items: each is a ChatTemplate [user_turn, assistant_turn]
        self.items: list[tuple[str, str]] = []
        for i in range(len(ds)):
            chat = ds[i]
            # Extract kalamang from user message and english from assistant
            kalamang = ""
            english = ""
            for turn in chat:
                if turn.role == "user":
                    # Extract after "Kalamang: "
                    msg = turn.message
                    if "Kalamang: " in msg:
                        kalamang = msg.split("Kalamang: ", 1)[1]
                elif turn.role == "assistant":
                    english = turn.message
            if kalamang and english:
                self.items.append((kalamang, english))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "mtob"

    def max_new_tokens(self, inference: Any) -> int:
        return 100

    def get(self, indx: int) -> Tuple[str, str]:
        kalamang, english = self.items[indx]
        prompt = encode_chat_template(
            template_kgv_to_en(kalamang),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, english

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
        return assistant_msgs[0].strip().lower()

    def score(self, ys: list[str], y_hats: list[str]) -> float:
        """Use exact match for simplicity; chrF could be added later."""
        results = [self.check(y, y_hat) for y, y_hat in zip(ys, y_hats)]
        return sum(results) / len(results)

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
