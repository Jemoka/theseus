from datasets import load_dataset
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(premise: str, hypothesis: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Does the hypothesis entail the premise? Please respond only with "entailment", "contradiction", or "neutral", not including quotes.

premise: {premise}
hypothesis: {hypothesis}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


class MNLI(ChatTemplateDataset):
    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("nyu-mll/multi_nli", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]

        if item["label"] == 0:
            answer = "entailment"
        elif item["label"] == 1:
            answer = "neutral"
        elif item["label"] == 2:
            answer = "contradiction"

        return template(premise, hypothesis, answer)
