from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset


class PG19(StreamingPretrainingDataset):
    """Project Gutenberg books (sedthh/gutenberg_english).

    48k+ English books from Project Gutenberg with metadata removed,
    suitable for long-context pretraining.
    """

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset(
            "sedthh/gutenberg_english", split="train", streaming=True
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            text = item.get("TEXT", "")
            if text:
                yield text
