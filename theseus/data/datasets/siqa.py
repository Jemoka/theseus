from datasets import load_dataset
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(
    context: str, question: str, answerA: str, answerB: str, answerC: str, label: str
) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Given this context and question, judge the best answer. Answer in a single letter, "A", "B", or "C", without quotes.

context: {context}
question: {question}

A: {answerA}
B: {answerB}
C: {answerC}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


class SIQA(ChatTemplateDataset):
    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("lighteval/siqa", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        label = ["A", "B", "C"][item["label"] - 1]
        return template(
            item["context"],
            item["question"],
            item["answerA"],
            item["answerB"],
            item["answerC"],
            label,
        )
