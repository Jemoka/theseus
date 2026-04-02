"""Dictionary learning next-token accuracy evaluation.

Feeds each val sequence through the model and checks whether the argmax
prediction at the SEP position equals the correct result token.

See ``theseus.data.datasets.dictlearn`` for the full dataset description.
"""

from theseus.data.datasets.dictlearn import DictLearn
from theseus.evaluation import EncodingEvaluation
from theseus.registry import evaluation


@evaluation("dictlearn")
class DictLearnEval(EncodingEvaluation):
    """Accuracy evaluation on the dictlearn val split.

    Each sequence is ``f1 f2 ... fn START v1 SEP result``.
    The model must predict ``result`` (the last token) given everything before it.
    We check whether the last token of the decoded argmax prediction matches.
    """

    def __init__(self) -> None:
        self._ds = DictLearn(split="val")

    @property
    def name(self) -> str:
        return "dictlearn"

    def __len__(self) -> int:
        return len(self._ds)

    def get(self, indx: int) -> str:
        return self._ds[indx]

    def clean(self, y_hat: str) -> str:
        # The decoded prediction is a space-separated integer string.
        # We only care about the last token (prediction at the SEP position).
        parts = y_hat.strip().split(" ")
        return parts[-1] if parts else ""

    def check(self, x: str, y_hat: str) -> bool:
        # x is the full sequence; the last token is the expected result.
        expected = x.strip().split(" ")[-1]
        return expected == y_hat
