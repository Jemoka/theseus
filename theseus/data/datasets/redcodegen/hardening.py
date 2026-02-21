import jsonlines
from pathlib import Path
from typing import Tuple

from theseus.data.datasets import ChatTemplate, ChatTurn, ContrastiveChatTemplateDataset


def template(prompt: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="system",
            message="""You are a helpful assistant for generating code. Given the prompt, generate Python code that solves the task; as a reminder, you should be writing production code (i.e. disable any debugging traces, etc.) Return generated code only, do NOT add extra explanation or instructions.""",
        ),
        ChatTurn(role="user", message=prompt),
        ChatTurn(role="assistant", message=label),
    ]


class RCGHardeningDataset(ContrastiveChatTemplateDataset):
    def __init__(self, split: str = "noop", config: str = "") -> None:
        """Load the RedCodeGen repo generated contrastive learning dataset."""

        # we ignore split, but config contains a string path
        # wrt where the hardening contrastive data live
        # config = "/Users/houjun/Downloads/qwen-contrastive-2pairs.jsonl"
        # self = SimpleNamespace()

        config_path = Path(config).resolve(strict=True)
        with jsonlines.open(config_path) as d:
            self.raw = [i for i in d]

        all_pairs = []
        for i in self.raw:
            p = i["pairs"]
            for j in p:
                j["prompt"] = i["prompt"]
                all_pairs.append(j)

        self.dataset = all_pairs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[ChatTemplate, ChatTemplate]:
        sample = self.dataset[idx]
        y_pos = template(sample["prompt"], sample["success"])
        y_neg = template(sample["prompt"], sample["failure"])

        return (y_pos, y_neg)
