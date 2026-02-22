from typing import Any
from theseus.job import BasicJob

from .hardening import Hardening

JOBS: dict[str, type[BasicJob[Any]]] = {"redcodegen/train/hardening": Hardening}
