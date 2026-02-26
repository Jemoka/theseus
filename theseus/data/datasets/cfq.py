from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(question: str, query: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Translate the following natural language question into a "
                "SPARQL query. Respond with only the SPARQL query, nothing else.\n\n"
                f"Question: {question}"
            ),
        ),
        ChatTurn(role="assistant", message=query),
    ]


class CFQ(ChatTemplateDataset):
    """Compositional Freebase Questions (Google).

    Each example maps a natural language question to a SPARQL query.
    The ``config`` parameter selects the MCD split (default ``"mcd1"``).
    """

    def __init__(self, split: str = "train", config: str | None = "mcd1") -> None:
        mcd = config or "mcd1"
        self.ds = load_dataset("google-research-datasets/cfq", mcd, split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["question"], item["query"])
