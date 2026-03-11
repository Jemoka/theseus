"""
Perplexity evaluation on WildChat validation data.
"""

from datasets import load_dataset

from theseus.evaluation import PerplexityEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import encode_chat_template, get_tokenizer
from theseus.data.datasets import ChatTurn


@evaluation("wildchat_ppl")
class WildChatPerplexityEval(PerplexityEvaluation):
    """Perplexity on WildChat conversations (assistant turns only)."""

    def __init__(self, num_samples: int = 500) -> None:
        ds = load_dataset("allenai/WildChat-1M", split="train")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "wildchat_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        chat = []
        for msg in item["conversation"]:
            chat.append(ChatTurn(role=msg["role"], message=msg["content"]))
        return encode_chat_template(chat, self.encoder, tokenize=False)
