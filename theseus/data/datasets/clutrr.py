from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(story: str, query: str, target_text: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following story about a family and determine the "
                "relationship between the two people mentioned in the query. "
                "Respond with only the relationship (e.g. aunt, grandfather, "
                "brother, sister, father, mother, etc.).\n\n"
                f"Story: {story}\n\n"
                f"Query: {query}"
            ),
        ),
        ChatTurn(role="assistant", message=target_text),
    ]


class CLUTRR(ChatTemplateDataset):
    """CLUTRR relational reasoning benchmark (Facebook Research).

    Given a semi-synthetic story about a hypothetical family, infer the
    kinship relation between two specified family members.  The ``config``
    parameter selects the subset (default ``"gen_train234_test2to10"``).
    """

    def __init__(
        self,
        split: str = "train",
        config: str | None = "gen_train234_test2to10",
    ) -> None:
        subset = config or "gen_train234_test2to10"
        self.ds = load_dataset(
            "parquet",
            data_files=(
                f"hf://datasets/CLUTRR/v1@refs/convert/parquet/"
                f"{subset}/{split}/*.parquet"
            ),
            split="train",
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["story"], item["query"], item["target_text"])
