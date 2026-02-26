"""
peS2o perplexity evaluation.

Measures language model perplexity on scientific papers from the peS2o corpus.
Returns 1/perplexity (higher is better).
"""

from datasets import load_dataset

from theseus.evaluation import PerplexityEvaluation


class Pes2OEval(PerplexityEvaluation):
    """Perplexity evaluation on peS2o scientific papers."""

    def __init__(self, num_samples: int = 500) -> None:
        ds = load_dataset(
            "BEE-spoke-data/peS2o-100k_en-xlong",
            split="train",
            streaming=True,
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
        return "pes2o"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
