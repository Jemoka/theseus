from typing import Callable

from theseus.evaluation.datasets.blimp import Blimp
from theseus.evaluation.datasets.fever import FEVEREval
from theseus.evaluation.datasets.longbench import LongBench
from theseus.evaluation.datasets.mnli import MNLIEval
from theseus.evaluation.datasets.qqp import QQPEval
from theseus.evaluation.datasets.siqa import SIQAEval
from theseus.evaluation.datasets.sst2 import SST2Eval
from theseus.evaluation.datasets.winogrande import WinograndeEval
from theseus.evaluation.base import Evaluation

DATASETS: dict[str, Callable[[], Evaluation]] = {
    "blimp": Blimp,
    "fever": FEVEREval,
    "longbench": LongBench,
    "mnli": MNLIEval,
    "qqp": QQPEval,
    "siqa": SIQAEval,
    "sst2": SST2Eval,
    "winogrande": WinograndeEval,
}
