from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset


class Pes2O(StreamingPretrainingDataset):
    """Scientific papers from the peS2o corpus (AllenAI).

    Streams full-text academic papers derived from the Semantic Scholar
    Open Research Corpus.  Uses ``allenai/peS2o`` via parquet auto-convert
    to bypass deprecated custom loading scripts.
    """

    def __init__(
        self,
        config: str | None = None,
        split: str = "train",
    ) -> None:
        version = config or "v2"
        self.ds = load_dataset(
            "parquet",
            data_files=(
                f"hf://datasets/allenai/peS2o@refs/convert/parquet/"
                f"{version}/partial-train/*.parquet"
            ),
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            text = item.get("text", "")
            if text:
                yield text
