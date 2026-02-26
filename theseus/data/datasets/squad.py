from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(context: str, question: str, answer: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following passage and answer the question. "
                "Respond with only the answer extracted from the passage.\n\n"
                f"Passage: {context}\n\n"
                f"Question: {question}"
            ),
        ),
        ChatTurn(role="assistant", message=answer),
    ]


class SQuAD(ChatTemplateDataset):
    """SQuAD v1.1 (Rajpurkar et al., 2016).

    Extractive question answering over Wikipedia paragraphs.
    """

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("rajpurkar/squad", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        return template(item["context"], item["question"], answer)
