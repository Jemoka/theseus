from collections.abc import Iterator
from datasets import load_dataset
from theseus.data.datasets import StreamingPretrainingDataset


class FineWeb(StreamingPretrainingDataset):
    def __init__(self, snapshot: str = "CC-MAIN-2022-21") -> None:
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb", snapshot, split="train", streaming=True
        )

    def __iter__(self) -> Iterator[str]:
        it: Iterator[str] = self.ds["text"]
        return it
