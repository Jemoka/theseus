"""
Cluster information.
"""

from typing import Annotated
from pydantic import BaseModel, Field, model_validator

from theseus.models.chip import Chip


class Cluster(BaseModel):
    name: str


class HardwareRequest(BaseModel):
    """
    Minimal, intent-level hardware request.

    Assumptions:
      - Storage is uniform across hosts (distributed FS).
      - Interconnects / topology are cluster concerns.
    """

    chip: Annotated[
        Chip,
        Field(description="chip type"),
    ]
    min_chips: Annotated[
        int,
        Field(ge=1, description="minimum number of chips requested"),
    ]
    preferred_hosts: Annotated[
        list[Cluster],
        Field(default_factory=list, description="hosts to prefer"),
    ]
    forbidden_hosts: Annotated[
        list[str],
        Field(default_factory=list, description="hosts to avoid"),
    ]

    @model_validator(mode="after")
    def _validate(self) -> "HardwareRequest":
        overlap = set(self.preferred_hosts) & set(self.forbidden_hosts)
        if overlap:
            raise ValueError(
                f"Hosts cannot be both preferred and forbidden: {sorted([i.name for i in overlap])}"
            )

        return self
