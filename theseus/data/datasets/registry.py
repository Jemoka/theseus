from theseus.data.datasets.alpaca import Alpaca
from theseus.data.datasets.bbq import BBQ
from theseus.data.datasets.ccaligned import CCAligned
from theseus.data.datasets.cfq import CFQ
from theseus.data.datasets.clutrr import CLUTRR
from theseus.data.datasets.fever import FEVER
from theseus.data.datasets.fineweb import FineWeb
from theseus.data.datasets.harmfulqa import HarmfulQA
from theseus.data.datasets.longbench import LongBench
from theseus.data.datasets.longhealth import LongHealth
from theseus.data.datasets.mmlu import MMLU
from theseus.data.datasets.mnli import MNLI
from theseus.data.datasets.mtob import MTOB
from theseus.data.datasets.pes2o import Pes2O
from theseus.data.datasets.pg19 import PG19
from theseus.data.datasets.pile import Pile
from theseus.data.datasets.pile_detoxify import PileDetoxify
from theseus.data.datasets.qqp import QQP
from theseus.data.datasets.siqa import SIQA
from theseus.data.datasets.squad import SQuAD
from theseus.data.datasets.sst2 import SST2
from theseus.data.datasets.winogrande import Winogrande

from theseus.data.datasets.redcodegen.hardening import RCGHardeningDataset

DATASETS = {
    "alpaca": Alpaca,
    "bbq": BBQ,
    "ccaligned": CCAligned,
    "cfq": CFQ,
    "clutrr": CLUTRR,
    "fever": FEVER,
    "fineweb": FineWeb,
    "harmfulqa": HarmfulQA,
    "longbench": LongBench,
    "longhealth": LongHealth,
    "mmlu": MMLU,
    "mnli": MNLI,
    "mtob": MTOB,
    "pes2o": Pes2O,
    "pg19": PG19,
    "pile": Pile,
    "pile_detoxify": PileDetoxify,
    "qqp": QQP,
    "siqa": SIQA,
    "squad": SQuAD,
    "sst2": SST2,
    "winogrande": Winogrande,
    # project-specific datasets
    "redcodegen__hardening": RCGHardeningDataset,
}
