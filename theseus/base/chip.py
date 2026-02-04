"""
Chip information.
"""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field


class Chip(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(description="name of hardware")]
    display_name: Annotated[str, Field(description="display name for logs")]
    memory: Annotated[int, Field(description="bytes of memory per chip")]


SUPPORTED_CHIPS: dict[str, Chip] = {
    "cpu": Chip(
        name="cpu",
        display_name="CPU",
        memory=int(32 * 1024**3),
    ),
    "gb10": Chip(
        name="gb10",
        display_name="Nvidia GB10",
        memory=int(64 * 1024**3),
    ),
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
    "a6000": Chip(
        name="a6000",
        display_name="Nvidia RTX A6000",
        memory=int(48 * 1024**3),
    ),
    "ada6000": Chip(
        name="ada6000",
        display_name="Nvidia RTX A6000",
        memory=int(48 * 1024**3),
    ),
    "l40": Chip(
        name="l40",
        display_name="Nvidia L40",
        memory=int(48 * 1024**3),
    ),
    "l40s": Chip(
        name="l40s",
        display_name="Nvidia L40S",
        memory=int(48 * 1024**3),
    ),
    # TPUs
    "tpu-v2": Chip(
        name="tpu-v2",
        display_name="Google TPU v2",
        memory=int(8 * 1024**3),
    ),
    "tpu-v3": Chip(
        name="tpu-v3",
        display_name="Google TPU v3",
        memory=int(16 * 1024**3),
    ),
    "tpu-v4": Chip(
        name="tpu-v4",
        display_name="Google TPU v4",
        memory=int(32 * 1024**3),
    ),
    "tpu-v5e": Chip(
        name="tpu-v5e",
        display_name="Google TPU v5e",
        memory=int(16 * 1024**3),
    ),
    "tpu-v5p": Chip(
        name="tpu-v5p",
        display_name="Google TPU v5p",
        memory=int(95 * 1024**3),
    ),
}
