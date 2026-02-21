from datasets import load_dataset
from theseus.data.datasets import ChatTemplateDataset, ChatTemplate, ChatTurn


def template(q1: str, q2: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Are these two questions paraphrases of each other? Answer with "yes" or "no", not including quotes.

question 1: {q1}
question 2: {q2}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


class QQP(ChatTemplateDataset):
    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("nyu-mll/glue", "qqp", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(
            item["question1"], item["question2"], "yes" if item["label"] else "no"
        )
