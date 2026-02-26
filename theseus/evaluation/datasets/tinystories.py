"""
TinyStories perplexity evaluation.

Measures language model perplexity on the TinyStories validation set.
Returns 1/perplexity (higher is better).
"""

from theseus.evaluation import PerplexityEvaluation


class TinyStoriesEval(PerplexityEvaluation):
    """Perplexity evaluation on TinyStories validation stories.

    Loads from roneneldan/TinyStories on HuggingFace.
    """

    def __init__(self, num_samples: int = 500):
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))

    @property
    def name(self) -> str:
        return "tinystories"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        return self.ds[indx]["text"]  # type: ignore
