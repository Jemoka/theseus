"""
PG-19 (Gutenberg) perplexity evaluation.

Measures language model perplexity on Gutenberg books.
Returns 1/perplexity (higher is better).
"""

from datasets import load_dataset

from theseus.evaluation import PerplexityEvaluation


class PG19Eval(PerplexityEvaluation):
    """Perplexity evaluation on Project Gutenberg books."""

    def __init__(self, num_samples: int = 100) -> None:
        ds = load_dataset(
            "sedthh/gutenberg_english", split="train", streaming=True
        )
        self.items: list[str] = []
        for item in ds:
            text = item.get("TEXT", "")
            if text:
                self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "pg19"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
