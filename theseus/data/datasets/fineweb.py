from collections.abc import Iterator
from datasets import load_dataset
from theseus.data.datasets import StreamingPretrainingDataset


class FineWeb(StreamingPretrainingDataset):
    def __init__(
        self, snapshot: str = "CC-MAIN-2022-21", config: str | None = None
    ) -> None:
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb", snapshot, split="train", streaming=True
        )

    def __iter__(self) -> Iterator[str]:
        for i in self.ds["text"]:
            yield i
