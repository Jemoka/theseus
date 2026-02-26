"""
CCAligned perplexity evaluation.

Measures language model perplexity on multilingual parallel text.
Returns 1/perplexity (higher is better).
"""

from theseus.data.datasets.ccaligned import CCAligned
from theseus.evaluation import PerplexityEvaluation


class CCAlignedEval(PerplexityEvaluation):
    """Perplexity evaluation on CCAligned multilingual sentence pairs."""

    def __init__(self, num_samples: int = 500, lang: str = "fr_XX") -> None:
        self.lang = lang
        ds = CCAligned(config=lang)
        self.items: list[str] = []
        for text in ds:
            self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "ccaligned"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
