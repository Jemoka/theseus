from datasets import load_dataset
from typing import Any, Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


def template(story: str, query: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following story about a family and determine the "
                "relationship between the two people mentioned in the query. "
                "Respond with only the relationship (e.g. aunt, grandfather, "
                "brother, sister, father, mother, etc.).\n\n"
                f"Story: {story}\n\n"
                f"Query: {query}"
            ),
        ),
    ]


class CLUTRREval(RolloutEvaluation):
    """CLUTRR relational reasoning evaluation using test split."""

    def __init__(
        self,
        split: str = "test",
        config: str = "gen_train234_test2to10",
    ) -> None:
        self.ds = load_dataset(
            "parquet",
            data_files=(
                f"hf://datasets/CLUTRR/v1@refs/convert/parquet/"
                f"{config}/{split}/*.parquet"
            ),
            split="train",
        )
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "clutrr"

    def max_new_tokens(self, inference: Any) -> int:
        return 20

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        prompt = encode_chat_template(
            template(item["story"], item["query"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["target_text"]

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
