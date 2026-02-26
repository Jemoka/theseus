"""
CCAligned perplexity evaluation.

Measures language model perplexity on multilingual parallel text.
Returns 1/perplexity (higher is better).

Provides per-language eval classes so each can be tracked independently
across continual learning phases.
"""

from theseus.data.datasets.ccaligned import CCAligned
from theseus.evaluation import PerplexityEvaluation


class CCAlignedEval(PerplexityEvaluation):
    """Perplexity evaluation on CCAligned multilingual sentence pairs."""

    _lang: str = "fr_XX"

    def __init__(self, num_samples: int = 500) -> None:
        ds = CCAligned(config=self._lang)
        self.items: list[str] = []
        for text in ds:
            self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return f"ccaligned_{self._lang.lower()}"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]


class CCAlignedFrEval(CCAlignedEval):
    """CCAligned perplexity: English-French."""

    _lang = "fr_XX"


class CCAlignedDeEval(CCAlignedEval):
    """CCAligned perplexity: English-German."""

    _lang = "de_DE"


class CCAlignedZhEval(CCAlignedEval):
    """CCAligned perplexity: English-Chinese."""

    _lang = "zh_CN"
