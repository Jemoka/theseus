"""
Cluster information.
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from theseus.base.chip import Chip


class Cluster(BaseModel):
    name: str
    root: str  # root directory of checkpoints, code, etc.
    work: str  # work directory (where mirrors will be copied)

    @property
    def root_dir(self) -> Path:
        # panic if not exist
        root_dir = Path(self.root)
        if not root_dir.exists():
            raise ValueError(
                f"Cluster root directory does not exist: cluster={self.name}, root={root_dir}"
            )

        return root_dir

    @property
    def data_dir(self) -> Path:
        # make directory if not exist
        data_dir = Path(self.root_dir) / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def checkpoints_dir(self) -> Path:
        # make directory if not exist
        checkpoints_dir = Path(self.root_dir) / "checkpoints"
        if not checkpoints_dir.exists():
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return checkpoints_dir


class ClusterMachine(BaseModel):
    name: str
    cluster: Cluster
    resources: dict[Chip, int]


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
        list[ClusterMachine],
        Field(default_factory=list, description="hosts to prefer"),
    ]
    forbidden_hosts: Annotated[
        list[ClusterMachine],
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


class HardwareResult(BaseModel):
    """
    Concrete hardware allocation result.
    """

    chip: Annotated[
        Optional[Chip],
        Field(description="chip type"),
    ]
    hosts: Annotated[
        list[ClusterMachine],
        Field(description="allocated hosts"),
    ]
    total_chips: Annotated[
        int,
        Field(ge=0, description="total number of allocated chips"),
    ]


def _detect_local_gpus() -> tuple[Optional[Chip], int]:
    """
    Detect local GPUs using nvidia-smi and CUDA_VISIBLE_DEVICES.
    Returns (chip, count) or (None, 0) if no GPUs found.
    """
    import os
    import subprocess

    from theseus.base.chip import SUPPORTED_CHIPS

    # Check which devices are visible
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        if cuda_visible.strip() == "":
            return None, 0
        visible_indices = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_indices = None  # All GPUs visible

    # Query nvidia-smi for GPU info
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, 0

    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    if not lines:
        return None, 0

    # Parse GPU info and filter by visible indices
    gpus: list[tuple[int, str, int]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, mem_mb = int(parts[0]), parts[1], int(parts[2])
            if visible_indices is None or idx in visible_indices:
                gpus.append((idx, name, mem_mb))

    if not gpus:
        return None, 0

    # Match first GPU name to a supported chip
    _, gpu_name, gpu_mem_mb = gpus[0]
    gpu_name_lower = gpu_name.lower()

    matched_chip: Optional[Chip] = None
    for chip_key, chip in SUPPORTED_CHIPS.items():
        # Match by checking if chip name appears in the GPU name
        if chip_key.split("-")[0] in gpu_name_lower:
            # Refine match by memory if multiple variants exist
            chip_mem_gb = chip.memory // (1024**3)
            if abs(gpu_mem_mb // 1024 - chip_mem_gb) < 10:
                matched_chip = chip
                break
            elif matched_chip is None:
                matched_chip = chip

    if matched_chip is None:
        # Create a chip entry for unknown GPU
        matched_chip = Chip(
            name=gpu_name_lower.replace(" ", "-"),
            display_name=gpu_name,
            memory=gpu_mem_mb * 1024 * 1024,
        )

    return matched_chip, len(gpus)


def local(root_dir: str, work_dir: str) -> HardwareResult:
    chip, total_chips = _detect_local_gpus()

    resources: dict[Chip, int] = {}
    if chip is not None:
        resources[chip] = total_chips

    local_machine = ClusterMachine(
        name="local",
        cluster=Cluster(name="local", root=root_dir, work=work_dir),
        resources=resources,
    )
    return HardwareResult(chip=chip, hosts=[local_machine], total_chips=total_chips)
