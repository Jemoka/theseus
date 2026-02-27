from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn


def template(
    context: str, question: str, choices: list[str], answer: str
) -> ChatTemplate:
    choices_text = "\n".join(f"{chr(65 + i)}: {c}" for i, c in enumerate(choices))
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following context and answer the question by "
                "selecting the correct option.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, or C):"
            ),
        ),
        ChatTurn(role="assistant", message=answer),
    ]


class BBQ(ChatTemplateDataset):
    """BBQ: Bias Benchmark for QA (Parrish et al., 2022).

    Probes social biases across multiple demographic dimensions.
    The ``config`` parameter selects the bias category (default ``"all"``).
    Available configs: Age, Disability_status, Gender_identity, Nationality,
    Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender, Religion,
    SES, Sexual_orientation, all.
    """

    def __init__(self, split: str = "test", config: str | None = "all") -> None:
        category = config or "all"
        self.ds = load_dataset("lighteval/bbq_helm", category, split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        choices: list[str] = item["choices"]
        gold_idx: int = item["gold_index"]
        answer = chr(65 + gold_idx)
        return template(item["context"], item["question"], choices, answer)
