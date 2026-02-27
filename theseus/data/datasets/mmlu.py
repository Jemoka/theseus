from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(question: str, choices: list[str], answer: str) -> ChatTemplate:
    choices_text = "\n".join(f"{chr(65 + i)}: {c}" for i, c in enumerate(choices))
    return [
        ChatTurn(
            role="user",
            message=(
                "Answer the following multiple-choice question by selecting "
                "the correct option.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, C, or D):"
            ),
        ),
        ChatTurn(role="assistant", message=answer),
    ]


class MMLU(ChatTemplateDataset):
    """MMLU: Massive Multitask Language Understanding (Hendrycks et al., 2021).

    Multiple-choice questions across 57 subjects. The ``config``
    parameter selects a subject (default ``"all"``).
    """

    def __init__(self, split: str = "test", config: str | None = "all") -> None:
        subject = config or "all"
        self.ds = load_dataset("cais/mmlu", subject, split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        choices: list[str] = item["choices"]
        answer = chr(65 + int(item["answer"]))
        return template(item["question"], choices, answer)
