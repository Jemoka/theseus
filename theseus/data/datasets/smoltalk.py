from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


@dataset("smoltalk")
class SmolTalk(ChatTemplateDataset):
    """SmolTalk (HuggingFace, 2024). Synthetic multi-turn conversations.

    Alternative training data for DMA-CoT side-channel format.
    """

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("HuggingFaceTB/smoltalk", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        turns = []
        for msg in item["messages"]:
            role = msg["role"]
            content = msg["content"]
            turns.append(ChatTurn(role=role, message=content))
        return turns
