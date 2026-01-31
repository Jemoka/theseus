from theseus.config import build, field, configuration, configure

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Chicken:
    name: str = field("data/dataset")
    chob: str = field("chombai")


@dataclass
class TokenizeDatasetConfigBase:
    """Base config for dataset tokenization"""

    name: str = field("data/dataset")
    suffix: str = field("data/suffix", default="")
    val_pct: float = field("data/val_pct", default=0.05)
    seed: int = field("data/seed", default=2357)
    dataset: List[Tuple[str, float]] = field(
        "data/thing", default_factory=lambda: [("fineweb", 1.0)]
    )


res = TokenizeDatasetConfigBase("te")
cfg = build(res)
cfg.data.dataset = "12"


def make_thing():
    return configure(TokenizeDatasetConfigBase)


with configuration(cfg):
    thing = make_thing()

thing
cfg
