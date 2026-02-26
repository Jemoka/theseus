from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset


class Pes2O(StreamingPretrainingDataset):
    """Scientific papers from the peS2o corpus (AllenAI).

    Streams full-text academic papers derived from the Semantic Scholar
    Open Research Corpus.  Uses ``BEE-spoke-data/peS2o-100k_en-xlong``
    which is a cleaned English subset in standard parquet format.
    """

    def __init__(
        self,
        config: str | None = None,
        split: str = "train",
    ) -> None:
        self.ds = load_dataset(
            "BEE-spoke-data/peS2o-100k_en-xlong",
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            text = item.get("text", "")
            if text:
                yield text
