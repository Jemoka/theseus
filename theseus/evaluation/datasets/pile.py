"""
Pile perplexity evaluation.

Measures language model perplexity on a sample from the Pile.
Returns 1/perplexity (higher is better).
"""

from datasets import load_dataset

from theseus.evaluation import PerplexityEvaluation


class PileEval(PerplexityEvaluation):
    """Perplexity evaluation on the Pile (uncopyrighted)."""

    def __init__(self, num_samples: int = 500) -> None:
        ds = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )
        self.items: list[str] = []
        for item in ds:
            text = item.get("text", "")
            if text:
                self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "pile"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
