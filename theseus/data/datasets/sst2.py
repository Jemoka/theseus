from datasets import load_dataset
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(sentence: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Classify the sentiment of the following sentence; respond with "positive" or "negative", not including quotes.

sentence: {sentence}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


class SST2(ChatTemplateDataset):
    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("stanfordnlp/sst2", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        label = "positive" if item["label"] == 1 else "negative"
        return template(item["sentence"], label)
