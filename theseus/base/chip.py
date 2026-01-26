"""
Chip information.
"""

from typing import Annotated
from pydantic import BaseModel, Field


class Chip(BaseModel):
    name: Annotated[str, Field(description="name of hardware")]
    display_name: Annotated[str, Field(description="display name for logs")]
    memory: Annotated[int, Field(description="bytes of memory per chip")]


SUPPORTED_CHIPS: dict[str, Chip] = {
    "h200": Chip(
        name="h200",
        display_name="Nvidia H200",
        memory=int(143.8 * 1024**3),
    ),
    "h100": Chip(
        name="h100",
        display_name="Nvidia H100",
        memory=int(80 * 1024**3),
    ),
    "a100-sxm4-80gb": Chip(
        name="a100-sxm4-80gb",
        display_name="Nvidia A100 SXM4 80GB",
        memory=int(80 * 1024**3),
    ),
    "a100-pcie-40gb": Chip(
        name="a100-pcie-40gb",
        display_name="Nvidia A100 PCIe 40GB",
        memory=int(40 * 1024**3),
    ),
}
