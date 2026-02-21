from theseus.data.datasets.fever import FEVER
from theseus.data.datasets.fineweb import FineWeb
from theseus.data.datasets.longbench import LongBench
from theseus.data.datasets.mnli import MNLI
from theseus.data.datasets.qqp import QQP
from theseus.data.datasets.siqa import SIQA
from theseus.data.datasets.sst2 import SST2
from theseus.data.datasets.winogrande import Winogrande

from theseus.data.datasets.redcodegen.hardening import RCGHardeningDataset

DATASETS = {
    "fever": FEVER,
    "fineweb": FineWeb,
    "longbench": LongBench,
    "mnli": MNLI,
    "qqp": QQP,
    "siqa": SIQA,
    "sst2": SST2,
    "winogrande": Winogrande,
    # project-specific datasets
    "redcodegen__hardening": RCGHardeningDataset,
}
