from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset


class Pile(StreamingPretrainingDataset):
    """The Pile (uncopyrighted mirror) for general pretraining.

    Streams text from ``monology/pile-uncopyrighted``, an 825 GiB diverse
    open-source language modelling dataset with copyrighted subsets removed.
    """

    def __init__(
        self,
        config: str | None = None,
        split: str = "train",
    ) -> None:
        self.ds = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            text = item.get("text", "")
            if text:
                yield text
