from .adamw import adamw, AdamWConfig
from .muon import muon, MuonConfig, scale_by_muon

OPTIMIZERS = {"adamw": (adamw, AdamWConfig), "muon": (muon, MuonConfig)}

__all__ = [
    "OPTIMIZERS",
    "adamw",
    "AdamWConfig",
    "muon",
    "MuonConfig",
    "scale_by_muon",
]
