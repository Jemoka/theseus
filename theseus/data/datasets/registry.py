from theseus.data.datasets.fineweb import FineWeb
from theseus.data.datasets.mnli import MNLI
from theseus.data.datasets.qqp import QQP
from theseus.data.datasets.siqa import SIQA
from theseus.data.datasets.sst2 import SST2
from theseus.data.datasets.winogrande import Winogrande

DATASETS = {
    "fineweb": FineWeb,
    "mnli": MNLI,
    "qqp": QQP,
    "siqa": SIQA,
    "sst2": SST2,
    "winogrande": Winogrande,
}
