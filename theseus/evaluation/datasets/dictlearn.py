"""Dictionary learning next-token accuracy evaluation.

Feeds each val sequence through the model and checks whether the argmax
prediction at the SEP position equals the correct result token.

See ``theseus.data.datasets.dictlearn`` for the full dataset description.

Registered variants mirror the dataset registry:
- ``dictlearn_16``, ``dictlearn_512`` (default, 64 values)
- ``dictlearn_{16,512}_v{32,64,128,256,512,1024}``
"""

from theseus.data.datasets.dictlearn import (
    N_VALUES_SWEEP,
    _registry as _ds_registry,
)
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


def _make_eval_cls(name: str, ds_cls: type, eval_name: str) -> type:
    return type(
        name,
        (_DictLearnEvalBase,),
        {"_ds_cls": ds_cls, "_eval_name": eval_name},
    )


_SEQ_LENGTHS = [16, 512]

for _sl in _SEQ_LENGTHS:
    # Default variant
    _default_name = f"dictlearn_{_sl}"
    _default_cls = _make_eval_cls(
        f"DictLearnEval{_sl}", _ds_registry[_default_name], _default_name
    )
    evaluation(_default_name)(_default_cls)

    # Sweep variants
    for _nv in N_VALUES_SWEEP:
        _variant_name = f"dictlearn_{_sl}_v{_nv}"
        _variant_cls = _make_eval_cls(
            f"DictLearnEval{_sl}V{_nv}",
            _ds_registry[_variant_name],
            _variant_name,
        )
        evaluation(_variant_name)(_variant_cls)
