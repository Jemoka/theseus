from typing import Callable

from theseus.evaluation.datasets.bbq import BBQEval
from theseus.evaluation.datasets.blimp import Blimp
from theseus.evaluation.datasets.ccaligned import CCALIGNED_EVALS
from theseus.evaluation.datasets.cfq import CFQEval
from theseus.evaluation.datasets.clutrr import CLUTRREval
from theseus.evaluation.datasets.fever import FEVEREval
from theseus.evaluation.datasets.longbench import LongBench
from theseus.evaluation.datasets.longhealth import LongHealthEval
from theseus.evaluation.datasets.mmlu import MMLUEval
from theseus.evaluation.datasets.mnli import MNLIEval
from theseus.evaluation.datasets.mtob import MTOBEval
from theseus.evaluation.datasets.pes2o import Pes2OEval
from theseus.evaluation.datasets.pg19 import PG19Eval
from theseus.evaluation.datasets.pile import PileEval
from theseus.evaluation.datasets.qqp import QQPEval
from theseus.evaluation.datasets.siqa import SIQAEval
from theseus.evaluation.datasets.squad import SQuADEval
from theseus.evaluation.datasets.sst2 import SST2Eval
from theseus.evaluation.datasets.tinystories import TinyStoriesEval
from theseus.evaluation.datasets.winogrande import WinograndeEval
from theseus.evaluation.datasets.perplexity_evals import (
    MNLIPerplexityEval,
    QQPPerplexityEval,
    SST2PerplexityEval,
    SIQAPerplexityEval,
    WinograndePerplexityEval,
    FineWebPerplexityEval,
)
from theseus.evaluation.base import Evaluation

DATASETS: dict[str, Callable[[], Evaluation]] = {
    "bbq": BBQEval,
    "blimp": Blimp,
    # CCAligned per-language evals: ccaligned_fr_xx, ccaligned_de_de, etc.
    **CCALIGNED_EVALS,
    "cfq": CFQEval,
    "clutrr": CLUTRREval,
    "fever": FEVEREval,
    "fineweb_ppl": FineWebPerplexityEval,
    "longbench": LongBench,
    "longhealth": LongHealthEval,
    "mmlu": MMLUEval,
    "mnli": MNLIEval,
    "mnli_ppl": MNLIPerplexityEval,
    "mtob": MTOBEval,
    "pes2o": Pes2OEval,
    "pg19": PG19Eval,
    "pile": PileEval,
    "qqp": QQPEval,
    "qqp_ppl": QQPPerplexityEval,
    "siqa": SIQAEval,
    "siqa_ppl": SIQAPerplexityEval,
    "squad": SQuADEval,
    "sst2": SST2Eval,
    "sst2_ppl": SST2PerplexityEval,
    "tinystories": TinyStoriesEval,
    "winogrande": WinograndeEval,
    "winogrande_ppl": WinograndePerplexityEval,
}
