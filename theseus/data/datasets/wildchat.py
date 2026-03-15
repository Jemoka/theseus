from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


@dataset("wildchat")
class WildChat(ChatTemplateDataset):
    """WildChat (Zhao et al., 2024). Multi-turn conversations from GPT users.

    1M real conversations. User turns become side-channel inputs,
    assistant turns become the thinking stream for DMA-CoT training.
    """

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("allenai/WildChat-1M", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        turns = []
        for msg in item["conversation"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                turns.append(ChatTurn(role="system", message=content))
            elif role == "user":
                turns.append(ChatTurn(role="user", message=content))
            elif role == "assistant":
                turns.append(ChatTurn(role="assistant", message=content))
        return turns
