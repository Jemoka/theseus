"""Dictionary learning next-token accuracy evaluation.

Feeds each val sequence through the model and checks whether the argmax
prediction at the SEP position equals the correct result token.

See ``theseus.data.datasets.dictlearn`` for the full dataset description.
"""

from theseus.data.datasets.dictlearn import DictLearn16, DictLearn512
from theseus.evaluation import EncodingEvaluation
from theseus.registry import evaluation


class _DictLearnEvalBase(EncodingEvaluation):
    """Base accuracy evaluation for dictlearn variants.

    Each sequence is ``f1 f2 ... fn START v1 SEP result``.
    The model must predict ``result`` (the last token) given everything before it.
    We check whether the last token of the decoded argmax prediction matches.
    """

    _ds_cls: type
    _eval_name: str

    def __init__(self) -> None:
        self._ds: list[str] = self._ds_cls(split="val")

    @property
    def name(self) -> str:
        return self._eval_name

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


@evaluation("dictlearn_16")
class DictLearnEval16(_DictLearnEvalBase):
    """Accuracy evaluation on the dictlearn_16 val split."""

    _ds_cls = DictLearn16
    _eval_name = "dictlearn_16"


@evaluation("dictlearn_512")
class DictLearnEval512(_DictLearnEvalBase):
    """Accuracy evaluation on the dictlearn_512 val split."""

    _ds_cls = DictLearn512
    _eval_name = "dictlearn_512"
