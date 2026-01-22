"""
Hardware information.
"""

from pydantic import BaseModel, Field
from typing import Annotated


class Chip(BaseModel):
    name: Annotated[str, Field(description="name of hardware")]
    display_name: Annotated[
        str, Field(description="display name for hardware for logs")
    ]
    memory: Annotated[int, Field(description="number of bytes of memory available")]


SUPPORTED_HARDWARE = {
    "h200": Chip(
        name="h200",
        display_name="Nvidia H200",
        memory=int(143.8 * 1024**3),  # 143.8 GB
    ),
    "h100": Chip(
        name="h100",
        display_name="Nvidia H100",
        memory=int(80 * 1024**3),  # 80 GB
    ),
    "a100-sxm4-80gb": Chip(
        name="a100-sxm4-80gb",
        display_name="Nvidia A100 SXM4 80GB",
        memory=int(80 * 1024**3),  # 80 GB
    ),
    "a100-pcie-40gb": Chip(
        name="a100-pcie-40gb",
        display_name="Nvidia A100 PCIe 40GB",
        memory=int(40 * 1024**3),  # 40 GB
    ),
}
